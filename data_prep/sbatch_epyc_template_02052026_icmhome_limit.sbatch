#!/bin/bash
#SBATCH --job-name=REPLACE_WITH_SLURMJOBNAME
#SBATCH --account=REPLACE_WITH_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=REPLACE_WITH_WALLTIME
#SBATCH --partition=REPLACE_WITH_PARTITION

# Chunk assignment comes from the array task id (submitted via --array=0-(N-1)%C).
ASSIGNED_CHUNK=${SLURM_ARRAY_TASK_ID}

ProjectName="REPLACE_WITH_PROJECTNAME"
ProjectDir="REPLACE_WITH_PROJECTDIR"
OutputDir="REPLACE_WITH_OUTPUTDIR"
SlurmJobName="REPLACE_WITH_SLURMJOBNAME"
sdfPath="REPLACE_WITH_SDFPATH"

nproc=16
effort_1=2.
ScoreThreshold=-18.
N_jobs=5

mkdir -p "${OutputDir}"
cd "${OutputDir}"

# Force ICM to use the same installation as the binary (so _dockScan macro resolves)
ICM_BIN="/project/katritch_223/icm-3.9-3a/icmng"
ICM_HOME="$(cd "$(dirname "${ICM_BIN}")" && pwd)"
export ICMHOME="${ICM_HOME}"
export ICMCONFIG="${ICM_HOME}/_config"

# Dock files based on Rank
files=($(ls $sdfPath/*inx | sed 's/.*Rank\([0-9]*\)_.*/\1 &/' | sort -n | awk '{print $2}'))
for ((ii=0; ii<=${#files[@]}-1; ii++))
do
   f="${files[$ii]}"
   inx_file=$f
   inx_name="${f##*/}"
   inx_name="${inx_name%.*}"

   j=$((${ii} % ${N_jobs}))
   if [ "${j}" -eq "${ASSIGNED_CHUNK}" ]
   then
        if [[ -f "./${inx_name}.finished" ]]
        then
            continue
        fi

        echo $(date) > ${inx_name}.started
        sleep 50

        /project/katritch_223/icm-3.9-3a/icmng -vlscluster \
            _dockScan ${ProjectDir}/${ProjectName} \
            input=${inx_file} \
            effort=${effort_1} \
            proc=${nproc} \
            -noStructFilter \
            scoreCutoff=${ScoreThreshold} \
            name=${inx_name} \
            > ${SlurmJobName}_${ii}_${inx_name}.log

        if grep -q "FINISHED at ligand" ${SlurmJobName}_${ii}_${inx_name}.log
        then
            echo $(date) > ${inx_name}.finished
        fi
   fi
done

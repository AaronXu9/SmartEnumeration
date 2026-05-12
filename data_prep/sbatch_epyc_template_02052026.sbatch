#!/bin/bash
#SBATCH --job-name=REPLACE_WITH_SLURMJOBNAME_REPLACE_WITH_ASSIGNEDCHUNK
#SBATCH --account=REPLACE_WITH_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=REPLACE_WITH_WALLTIME
#SBATCH --partition=REPLACE_WITH_PARTITION

# ======================
# Docking
# ======================

ASSIGNED_CHUNK=REPLACE_WITH_ASSIGNEDCHUNK

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

# files=($sdfPath/*inx)
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
        # if [[ -f "./${ProjectName}_${inx_name}.ob" && -f "./${inx_name}.finished" ]]
        # detects finish by *.finished file only
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

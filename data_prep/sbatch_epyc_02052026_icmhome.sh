#!/bin/bash

# =====================
# I/O
# =====================
while getopts f:p:P:s:n:v:t:e:A:q:o: flag
do
    case "${flag}" in
        f) sdfPath=${OPTARG};;
        p) ProjectName=${OPTARG};;
        P) ProjectDir=${OPTARG};;
        s) ScoreThreshold=${OPTARG};;
        n) N_jobs=${OPTARG};;
        v) IcmngPath=${OPTARG};;
        t) wall_time=${OPTARG};;
        e) effort_1=${OPTARG};;
        A) SlurmAccount=${OPTARG};;     # NEW
        q) SlurmPartition=${OPTARG};;   # NEW
        o) OutputDir=${OPTARG};;         # NEW
    esac
done

# =====================
# Defaults
# =====================
: ${wall_time:="47:55:00"}
: ${effort_1:=2.}
: ${SlurmAccount:="katritch_502"}
: ${SlurmPartition:="epyc-64"}
: ${OutputDir:="${sdfPath}"}   # default: same as input dir

echo "sdfPath: $sdfPath"
echo "ProjectName: $ProjectName"
echo "ProjectDir: $ProjectDir"
echo "OutputDir: $OutputDir"
echo "ScoreThreshold: $ScoreThreshold"
echo "N_jobs: $N_jobs"
echo "IcmngPath: $IcmngPath"
echo "wall_time: $wall_time"
echo "effort_1: $effort_1"
echo "SlurmAccount: $SlurmAccount"
echo "SlurmPartition: $SlurmPartition"

# =====================
# Slurm settings
# =====================
SlurmJobName="W"

# =====================
# Body
# =====================
N_DockingTasks=$(ls -1 ${sdfPath}/*.inx | wc -l)
N_dependency=0
[[ $N_DockingTasks -gt 400 ]] && ((N_dependency++))
[[ $N_DockingTasks -gt 800 ]] && ((N_dependency++))

assignedchunk=0
while [[ $assignedchunk -lt $N_jobs ]]
do
    sed "s#REPLACE_WITH_WALLTIME#${wall_time}#g" sbatch_epyc_template_02052026_icmhome.sbatch > batch_${assignedchunk}.slurm
    sed -i "s#REPLACE_WITH_ACCOUNT#${SlurmAccount}#g" batch_${assignedchunk}.slurm
    sed -i "s#REPLACE_WITH_PARTITION#${SlurmPartition}#g" batch_${assignedchunk}.slurm
    sed -i "s#REPLACE_WITH_ASSIGNEDCHUNK#${assignedchunk}#g" batch_${assignedchunk}.slurm
    sed -i "s#REPLACE_WITH_PROJECTNAME#${ProjectName}#g" batch_${assignedchunk}.slurm
    sed -i "s#REPLACE_WITH_PROJECTDIR#${ProjectDir}#g" batch_${assignedchunk}.slurm
    sed -i "s#REPLACE_WITH_OUTPUTDIR#${OutputDir}#g" batch_${assignedchunk}.slurm
    sed -i "s#REPLACE_WITH_SLURMJOBNAME#${SlurmJobName}#g" batch_${assignedchunk}.slurm
    sed -i "s#REPLACE_WITH_SDFPATH#${sdfPath}#g" batch_${assignedchunk}.slurm
    sed -i "s#N_jobs=5#N_jobs=${N_jobs}#g" batch_${assignedchunk}.slurm
    sed -i "s#effort_1=2.#effort_1=${effort_1}#g" batch_${assignedchunk}.slurm
    sed -i "s#-18.#${ScoreThreshold}#g" batch_${assignedchunk}.slurm
    sed -i "s#/project/katritch_223/icm-3.9-3a/icmng#${IcmngPath}#g" batch_${assignedchunk}.slurm

    script="batch_${assignedchunk}.slurm"
    j0=$(sbatch ${script} | cut -d' ' -f4)

    if [[ $N_dependency -eq 1 ]]; then
        sbatch --dependency=afterany:$j0 ${script}
    elif [[ $N_dependency -eq 2 ]]; then
        j1=$(sbatch --dependency=afterany:$j0 ${script} | cut -d' ' -f4)
        sbatch --dependency=afterany:$j1 ${script}
    fi

    ((++assignedchunk))
done

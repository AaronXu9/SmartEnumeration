#!/bin/bash

# =====================
# I/O
# =====================
# Same flags as sbatch_epyc_02052026_icmhome.sh, plus:
#   -c <concurrency>   max concurrent array tasks (default 4)
while getopts f:p:P:s:n:v:t:e:A:q:o:c: flag
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
        A) SlurmAccount=${OPTARG};;
        q) SlurmPartition=${OPTARG};;
        o) OutputDir=${OPTARG};;
        c) Concurrency=${OPTARG};;
    esac
done

# =====================
# Defaults
# =====================
: ${wall_time:="47:55:00"}
: ${effort_1:=2.}
: ${SlurmAccount:="katritch_502"}
: ${SlurmPartition:="epyc-64"}
: ${OutputDir:="${sdfPath}"}
: ${Concurrency:=4}

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
echo "Concurrency (max concurrent array tasks): $Concurrency"

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
echo "N_DockingTasks: $N_DockingTasks  N_dependency: $N_dependency"

# Generate ONE shared batch script; all array tasks read it with their own SLURM_ARRAY_TASK_ID
script="batch_array.slurm"
sed "s#REPLACE_WITH_WALLTIME#${wall_time}#g" sbatch_epyc_template_02052026_icmhome_limit.sbatch > ${script}
sed -i "s#REPLACE_WITH_ACCOUNT#${SlurmAccount}#g"     ${script}
sed -i "s#REPLACE_WITH_PARTITION#${SlurmPartition}#g" ${script}
sed -i "s#REPLACE_WITH_PROJECTNAME#${ProjectName}#g"  ${script}
sed -i "s#REPLACE_WITH_PROJECTDIR#${ProjectDir}#g"    ${script}
sed -i "s#REPLACE_WITH_OUTPUTDIR#${OutputDir}#g"      ${script}
sed -i "s#REPLACE_WITH_SLURMJOBNAME#${SlurmJobName}#g" ${script}
sed -i "s#REPLACE_WITH_SDFPATH#${sdfPath}#g"          ${script}
sed -i "s#N_jobs=5#N_jobs=${N_jobs}#g"                ${script}
sed -i "s#effort_1=2.#effort_1=${effort_1}#g"         ${script}
sed -i "s#-18.#${ScoreThreshold}#g"                   ${script}
sed -i "s#/project/katritch_223/icm-3.9-3a/icmng#${IcmngPath}#g" ${script}

array_range="0-$((N_jobs-1))%${Concurrency}"
echo "Array spec: --array=${array_range}"

j0=$(sbatch --array=${array_range} ${script} | awk '{print $NF}')
echo "Submitted array job: ${j0}"

if [[ $N_dependency -ge 1 ]]; then
    j1=$(sbatch --dependency=afterany:${j0} --array=${array_range} ${script} | awk '{print $NF}')
    echo "Chained resume job 1: ${j1}"
fi
if [[ $N_dependency -ge 2 ]]; then
    j2=$(sbatch --dependency=afterany:${j1} --array=${array_range} ${script} | awk '{print $NF}')
    echo "Chained resume job 2: ${j2}"
fi

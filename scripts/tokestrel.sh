#!/bin/bash

ROOT=../
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$2
export PYTHONPATH=$ROOT:$PYTHONPATH
CPUS_PER_TASK=${CPUS_PER_TASK:-4}

spring.submit run -n$1 -p spring_scheduler --gpu --job-name=$3 --cpus-per-task=${CPUS_PER_TASK} \
"python -m eod to_kestrel \
  --config=$cfg \
  --save_to=kestrel_model \
  2>&1 | tee log.tokestrel.$T.$(basename $cfg) "

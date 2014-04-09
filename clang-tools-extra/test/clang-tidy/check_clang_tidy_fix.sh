#!/bin/sh
#
# Run clang-tidy in fix mode and verify the result.

INPUT_FILE=$1
CHECK_TO_RUN=$2
TEMPORARY_FILE=$3.cpp

grep -Ev "// *[A-Z-]+:" ${INPUT_FILE} > ${TEMPORARY_FILE}
clang-tidy ${TEMPORARY_FILE} -fix --checks=${CHECK_TO_RUN} \
    --disable-checks="" -- --std=c++11
FileCheck -input-file=${TEMPORARY_FILE} ${INPUT_FILE}

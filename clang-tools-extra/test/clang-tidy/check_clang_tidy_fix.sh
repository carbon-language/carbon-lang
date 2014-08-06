#!/bin/sh
#
# Run clang-tidy in fix mode and verify the result.

INPUT_FILE=$1
CHECK_TO_RUN=$2
TEMPORARY_FILE=$3.cpp
shift 3

# Remove the contents of the CHECK lines to avoid CHECKs matching on themselves.
# We need to keep the comments to preserve line numbers while avoiding empty
# lines which could potentially trigger formatting-related checks.
sed 's#// *[A-Z-]\+:.*#//#' ${INPUT_FILE} > ${TEMPORARY_FILE}

clang-tidy ${TEMPORARY_FILE} -fix --checks="-*,${CHECK_TO_RUN}" -- --std=c++11 \
  $* > ${TEMPORARY_FILE}.msg 2>&1 || exit $?

FileCheck -input-file=${TEMPORARY_FILE} ${INPUT_FILE} \
  -check-prefix=CHECK-FIXES -strict-whitespace || exit $?

if grep -q CHECK-MESSAGES ${INPUT_FILE}; then
  FileCheck -input-file=${TEMPORARY_FILE}.msg ${INPUT_FILE} \
    -check-prefix=CHECK-MESSAGES -implicit-check-not="{{warning|error}}:" \
    || exit $?
fi

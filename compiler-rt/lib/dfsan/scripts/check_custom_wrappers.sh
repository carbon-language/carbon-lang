#!/bin/bash

DFSAN_DIR=$(dirname "$0")/../
DFSAN_CUSTOM_TESTS=${DFSAN_DIR}/../../test/dfsan/custom.c
DFSAN_CUSTOM_WRAPPERS=${DFSAN_DIR}/dfsan_custom.cc
DFSAN_ABI_LIST=${DFSAN_DIR}/done_abilist.txt

DIFFOUT=$(mktemp -q /tmp/tmp.XXXXXXXXXX)
ERRORLOG=$(mktemp -q /tmp/tmp.XXXXXXXXXX)

on_exit() {
  rm -f ${DIFFOUT} 2> /dev/null
  rm -f ${ERRORLOG} 2> /dev/null
}

trap on_exit EXIT

diff -u \
  <(grep -E "^fun:.*=custom" ${DFSAN_ABI_LIST} | grep -v "dfsan_get_label" \
    | sed "s/^fun:\(.*\)=custom.*/\1/" | sort ) \
  <(grep -E "__dfsw.*\(" ${DFSAN_CUSTOM_WRAPPERS} \
    | sed "s/.*__dfsw_\(.*\)(.*/\1/" \
    | sort) > ${DIFFOUT}
if [ $? -ne 0 ]
then
  echo -n "The following differences between the ABI list and ">> ${ERRORLOG}
  echo "the implemented custom wrappers have been found:" >> ${ERRORLOG}
  cat ${DIFFOUT} >> ${ERRORLOG}
fi

diff -u \
  <(grep -E __dfsw_ ${DFSAN_CUSTOM_WRAPPERS} \
    | sed "s/.*__dfsw_\([^(]*\).*/\1/" \
    | sort) \
  <(grep -E "^\\s*test_.*\(\);" ${DFSAN_CUSTOM_TESTS} \
    | sed "s/.*test_\(.*\)();/\1/" \
    | sort) > ${DIFFOUT}
if [ $? -ne 0 ]
then
  echo -n "The following differences between the implemented " >> ${ERRORLOG}
  echo "custom wrappers and the tests have been found:" >> ${ERRORLOG}
  cat ${DIFFOUT} >> ${ERRORLOG}
fi

if [[ -s ${ERRORLOG} ]]
then
  cat ${ERRORLOG}
  exit 1
fi


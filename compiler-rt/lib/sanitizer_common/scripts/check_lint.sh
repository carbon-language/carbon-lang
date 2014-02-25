#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Guess path to LLVM_CHECKOUT if not provided
if [ "${LLVM_CHECKOUT}" == "" ]; then
  LLVM_CHECKOUT="${SCRIPT_DIR}/../../../../../"
fi

# Cpplint setup
CPPLINT=${SCRIPT_DIR}/cpplint.py
if [ "${PYTHON_EXECUTABLE}" != "" ]; then
  CPPLINT="${PYTHON_EXECUTABLE} ${CPPLINT}"
fi

# Filters
# TODO: remove some of these filters
LLVM_LINT_FILTER=-,+whitespace
COMMON_LINT_FILTER=-build/include,-build/header_guard,-legal/copyright,-whitespace/comments,-readability/casting,\
-build/namespaces
ASAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/int
ASAN_TEST_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/sizeof,-runtime/int,-runtime/printf,-runtime/threadsafe_fn
ASAN_LIT_TEST_LINT_FILTER=${ASAN_TEST_LINT_FILTER},-whitespace/line_length
TSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER}
TSAN_TEST_LINT_FILTER=${TSAN_RTL_LINT_FILTER},-runtime/threadsafe_fn,-runtime/int
TSAN_LIT_TEST_LINT_FILTER=${TSAN_TEST_LINT_FILTER},-whitespace/line_length
MSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER}
LSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER}
LSAN_LIT_TEST_LINT_FILTER=${LSAN_RTL_LINT_FILTER},-whitespace/line_length
DFSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/int,-runtime/printf,-runtime/references
COMMON_RTL_INC_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/int,-runtime/sizeof,-runtime/printf,-readability/fn_size
SANITIZER_INCLUDES_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/int
MKTEMP="mktemp -q /tmp/tmp.XXXXXXXXXX"
cd ${LLVM_CHECKOUT}

EXITSTATUS=0
ERROR_LOG=$(${MKTEMP})

run_lint() {
  FILTER=$1
  shift
  TASK_LOG=$(${MKTEMP})
  ${CPPLINT} --filter=${FILTER} "$@" 2>$TASK_LOG
  if [ "$?" != "0" ]; then
    cat $TASK_LOG | grep -v "Done processing" | grep -v "Total errors found" \
      | grep -v "Skipping input" >> $ERROR_LOG
  fi
  if [[ "${SILENT}" != "1" ]]; then
    cat $TASK_LOG
  fi
}

run_lint ${LLVM_LINT_FILTER} --filter=${LLVM_LINT_FILTER} \
  lib/Transforms/Instrumentation/*Sanitizer.cpp \
  lib/Transforms/Utils/SpecialCaseList.cpp &

COMPILER_RT=projects/compiler-rt
LIT_TESTS=${COMPILER_RT}/test
# Headers
SANITIZER_INCLUDES=${COMPILER_RT}/include/sanitizer
run_lint ${SANITIZER_INCLUDES_LINT_FILTER} ${SANITIZER_INCLUDES}/*.h &

# Sanitizer_common
COMMON_RTL=${COMPILER_RT}/lib/sanitizer_common
run_lint ${COMMON_RTL_INC_LINT_FILTER} ${COMMON_RTL}/*.{cc,h} \
                                       ${COMMON_RTL}/tests/*.cc &

# Interception
INTERCEPTION=${COMPILER_RT}/lib/interception
run_lint ${ASAN_RTL_LINT_FILTER} ${INTERCEPTION}/*.{cc,h} &

# ASan
ASAN_RTL=${COMPILER_RT}/lib/asan
run_lint ${ASAN_RTL_LINT_FILTER} ${ASAN_RTL}/*.{cc,h} &
run_lint ${ASAN_TEST_LINT_FILTER} ${ASAN_RTL}/tests/*.{cc,h} &
run_lint ${ASAN_LIT_TEST_LINT_FILTER} ${LIT_TESTS}/asan/*/*.cc &

# TSan
TSAN_RTL=${COMPILER_RT}/lib/tsan
run_lint ${TSAN_RTL_LINT_FILTER} ${TSAN_RTL}/rtl/*.{cc,h} &
run_lint ${TSAN_TEST_LINT_FILTER} ${TSAN_RTL}/tests/rtl/*.{cc,h} \
                                  ${TSAN_RTL}/tests/unit/*.cc &
run_lint ${TSAN_LIT_TEST_LINT_FILTER} ${LIT_TESTS}/tsan/*.cc &

# MSan
MSAN_RTL=${COMPILER_RT}/lib/msan
run_lint ${MSAN_RTL_LINT_FILTER} ${MSAN_RTL}/*.{cc,h} &

# LSan
LSAN_RTL=${COMPILER_RT}/lib/lsan
run_lint ${LSAN_RTL_LINT_FILTER} ${LSAN_RTL}/*.{cc,h}
run_lint ${LSAN_LIT_TEST_LINT_FILTER} ${LIT_TESTS}/lsan/*/*.cc &

# DFSan
DFSAN_RTL=${COMPILER_RT}/lib/dfsan
run_lint ${DFSAN_RTL_LINT_FILTER} ${DFSAN_RTL}/*.{cc,h} &
${DFSAN_RTL}/scripts/check_custom_wrappers.sh >> $ERROR_LOG

# Misc files
FILES=${COMMON_RTL}/*.inc
TMPFILES=""
for FILE in $FILES; do
  TMPFILE="$(${MKTEMP}).$(basename ${FILE}).cc"
  cp -f $FILE $TMPFILE
  run_lint ${COMMON_RTL_INC_LINT_FILTER} $TMPFILE &
  TMPFILES="$TMPFILES $TMPFILE"
done

wait

for temp in $TMPFILES; do
  rm -f $temp
done

if [[ -s $ERROR_LOG ]]; then
  cat $ERROR_LOG
  exit 1
fi

exit 0

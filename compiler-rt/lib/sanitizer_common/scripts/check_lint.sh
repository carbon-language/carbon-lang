#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Guess path to LLVM_CHECKOUT if not provided
if [ "${LLVM_CHECKOUT}" == "" ]; then
  LLVM_CHECKOUT="${SCRIPT_DIR}/../../../../../"
fi

# Cpplint setup
cd ${SCRIPT_DIR}
CPPLINT=${SCRIPT_DIR}/cpplint.py

# Filters
# TODO: remove some of these filters
LLVM_LINT_FILTER=-,+whitespace
COMMON_LINT_FILTER=-build/include,-build/header_guard,-legal/copyright,-whitespace/comments,-readability/casting,\
-build/namespaces
ASAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/int
ASAN_TEST_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/sizeof,-runtime/int,-runtime/printf
ASAN_LIT_TEST_LINT_FILTER=${ASAN_TEST_LINT_FILTER},-whitespace/line_length
TSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER}
TSAN_TEST_LINT_FILTER=${TSAN_RTL_LINT_FILTER},-runtime/threadsafe_fn,-runtime/int
TSAN_LIT_TEST_LINT_FILTER=${TSAN_TEST_LINT_FILTER},-whitespace/line_length
MSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER}
LSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER}
LSAN_LIT_TEST_LINT_FILTER=${LSAN_RTL_LINT_FILTER},-whitespace/line_length
COMMON_RTL_INC_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/int,-runtime/sizeof,-runtime/printf
SANITIZER_INCLUDES_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/int

cd ${LLVM_CHECKOUT}

EXITSTATUS=0
LOG=$(mktemp -q)

run_lint() {
  FILTER=$1
  shift
  if [[ "${SILENT}" == "1" && "${LOG}" != "" ]]; then
    ${CPPLINT} --filter=${FILTER} "$@" 2>>$LOG
  else
    ${CPPLINT} --filter=${FILTER} "$@"
  fi
  if [ "$?" != "0" ]; then
    EXITSTATUS=1
  fi
}

run_lint ${LLVM_LINT_FILTER} --filter=${LLVM_LINT_FILTER} \
  lib/Transforms/Instrumentation/*Sanitizer.cpp \
  lib/Transforms/Utils/SpecialCaseList.cpp

COMPILER_RT=projects/compiler-rt
# Headers
SANITIZER_INCLUDES=${COMPILER_RT}/include/sanitizer
run_lint ${SANITIZER_INCLUDES_LINT_FILTER} ${SANITIZER_INCLUDES}/*.h

# Sanitizer_common
COMMON_RTL=${COMPILER_RT}/lib/sanitizer_common
run_lint ${COMMON_RTL_INC_LINT_FILTER} ${COMMON_RTL}/*.{cc,h} \
                                       ${COMMON_RTL}/tests/*.cc

# Interception
INTERCEPTION=${COMPILER_RT}/lib/interception
run_lint ${ASAN_RTL_LINT_FILTER} ${INTERCEPTION}/*.{cc,h}

# ASan
ASAN_RTL=${COMPILER_RT}/lib/asan
run_lint ${ASAN_RTL_LINT_FILTER} ${ASAN_RTL}/*.{cc,h}
run_lint ${ASAN_TEST_LINT_FILTER} ${ASAN_RTL}/tests/*.{cc,h}
run_lint ${ASAN_LIT_TEST_LINT_FILTER} ${ASAN_RTL}/lit_tests/*/*.cc

# TSan
TSAN_RTL=${COMPILER_RT}/lib/tsan
run_lint ${TSAN_RTL_LINT_FILTER} ${TSAN_RTL}/rtl/*.{cc,h}
run_lint ${TSAN_TEST_LINT_FILTER} ${TSAN_RTL}/tests/rtl/*.{cc,h} \
                                  ${TSAN_RTL}/tests/unit/*.cc
run_lint ${TSAN_LIT_TEST_LINT_FILTER} ${TSAN_RTL}/lit_tests/*.cc

# MSan
MSAN_RTL=${COMPILER_RT}/lib/msan
run_lint ${MSAN_RTL_LINT_FILTER} ${MSAN_RTL}/*.{cc,h}

# LSan
LSAN_RTL=${COMPILER_RT}/lib/lsan
run_lint ${LSAN_RTL_LINT_FILTER} ${LSAN_RTL}/*.{cc,h}
run_lint ${LSAN_RTL_LINT_FILTER} ${LSAN_RTL}/tests/*.{cc,h}
run_lint ${LSAN_LIT_TEST_LINT_FILTER} ${LSAN_RTL}/lit_tests/*/*.cc

# Misc files
FILES=${COMMON_RTL}/*.inc
for FILE in $FILES; do
    TMPFILE=$(mktemp -u ${FILE}.XXXXX).cc
    cp -f $FILE $TMPFILE && \
        run_lint ${COMMON_RTL_INC_LINT_FILTER} $TMPFILE
    rm $TMPFILE
done

if [ "$EXITSTATUS" != "0" ]; then
  cat $LOG | grep -v "Done processing" | grep -v "Total errors found" \
    grep -v "Skipping input"
fi

exit $EXITSTATUS

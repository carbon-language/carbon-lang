#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Guess path to LLVM_CHECKOUT if not provided
if [ "${LLVM_CHECKOUT}" == "" ]; then
  LLVM_CHECKOUT="${SCRIPT_DIR}/../../../../../"
  echo "LLVM Checkout: ${LLVM_CHECKOUT}"
fi

# Cpplint setup
cd ${SCRIPT_DIR}
if [ ! -d cpplint ]; then
  svn co http://google-styleguide.googlecode.com/svn/trunk/cpplint cpplint
else
  (cd cpplint && svn up)
fi
CPPLINT=${SCRIPT_DIR}/cpplint/cpplint.py

# Filters
# TODO: remove some of these filters
COMMON_LINT_FILTER=-build/include,-build/header_guard,-legal/copyright,-whitespace/comments,-readability/casting,\
-build/namespaces
ASAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER},-readability/check,-runtime/int
ASAN_TEST_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/sizeof,-runtime/int,-runtime/printf
ASAN_LIT_TEST_LINT_FILTER=${ASAN_TEST_LINT_FILTER},-whitespace/line_length
TSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER}
TSAN_TEST_LINT_FILTER=${TSAN_RTL_LINT_FILTER},-runtime/threadsafe_fn,-runtime/int
TSAN_LIT_TEST_LINT_FILTER=${TSAN_TEST_LINT_FILTER},-whitespace/line_length
MSAN_RTL_LINT_FILTER=${COMMON_LINT_FILTER}
COMMON_RTL_INC_LINT_FILTER=${COMMON_LINT_FILTER},-runtime/int,-runtime/sizeof,-runtime/printf

cd ${LLVM_CHECKOUT}

# LLVM Instrumentation
LLVM_INSTRUMENTATION=lib/Transforms/Instrumentation
LLVM_LINT_FILTER=-,+whitespace
${CPPLINT} --filter=${LLVM_LINT_FILTER} ${LLVM_INSTRUMENTATION}/*Sanitizer.cpp \
                                        ${LLVM_INSTRUMENTATION}/BlackList.*

COMPILER_RT=projects/compiler-rt

# Headers
SANITIZER_INCLUDES=${COMPILER_RT}/include/sanitizer
${CPPLINT} --filter=${TSAN_RTL_LINT_FILTER} ${SANITIZER_INCLUDES}/*.h

# Sanitizer_common
COMMON_RTL=${COMPILER_RT}/lib/sanitizer_common
${CPPLINT} --filter=${COMMON_RTL_INC_LINT_FILTER} ${COMMON_RTL}/*.{cc,h}
${CPPLINT} --filter=${COMMON_RTL_INC_LINT_FILTER} ${COMMON_RTL}/tests/*.cc

# Interception
INTERCEPTION=${COMPILER_RT}/lib/interception
${CPPLINT} --filter=${ASAN_RTL_LINT_FILTER} ${INTERCEPTION}/*.{cc,h}

# ASan
ASAN_RTL=${COMPILER_RT}/lib/asan
${CPPLINT} --filter=${ASAN_RTL_LINT_FILTER} ${ASAN_RTL}/*.{cc,h}
${CPPLINT} --filter=${ASAN_TEST_LINT_FILTER} ${ASAN_RTL}/tests/*.{cc,h}
${CPPLINT} --filter=${ASAN_LIT_TEST_LINT_FILTER} ${ASAN_RTL}/lit_tests/*.cc \
                                             ${ASAN_RTL}/lit_tests/*/*.cc \

# TSan
TSAN_RTL=${COMPILER_RT}/lib/tsan
${CPPLINT} --filter=${TSAN_RTL_LINT_FILTER} ${TSAN_RTL}/rtl/*.{cc,h}
${CPPLINT} --filter=${TSAN_TEST_LINT_FILTER} ${TSAN_RTL}/tests/rtl/*.{cc,h} \
                                             ${TSAN_RTL}/tests/unit/*.cc
${CPPLINT} --filter=${TSAN_LIT_TEST_LINT_FILTER} ${TSAN_RTL}/lit_tests/*.cc

# MSan
MSAN_RTL=${COMPILER_RT}/lib/msan
${CPPLINT} --filter=${MSAN_RTL_LINT_FILTER} ${MSAN_RTL}/*.{cc,h}

set +e

# Misc files
FILES=${COMMON_RTL}/*.inc
for FILE in $FILES; do
    TMPFILE=$(mktemp -u ${FILE}.XXXXX).cc
    echo "Checking $FILE"
    cp -f $FILE $TMPFILE && \
        ${CPPLINT} --filter=${COMMON_RTL_INC_LINT_FILTER} $TMPFILE
    rm $TMPFILE
done

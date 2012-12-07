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
  svn co -r83 http://google-styleguide.googlecode.com/svn/trunk/cpplint cpplint
fi
CPPLINT=${SCRIPT_DIR}/cpplint/cpplint.py

# Filters
# TODO: remove some of these filters
ASAN_RTL_LINT_FILTER=-readability/casting,-readability/check,-build/include,-build/header_guard,-build/class,-legal/copyright,-build/namespaces
ASAN_TEST_LINT_FILTER=-readability/casting,-build/include,-legal/copyright,-whitespace/newline,-runtime/sizeof,-runtime/int,-runtime/printf,-build/header_guard
TSAN_RTL_LINT_FILTER=-legal/copyright,-build/include,-readability/casting,-build/header_guard,-build/namespaces
TSAN_TEST_LINT_FILTER=${TSAN_RTL_LINT_FILTER},-runtime/threadsafe_fn,-runtime/int

cd ${LLVM_CHECKOUT}

# LLVM Instrumentation
LLVM_INSTRUMENTATION=lib/Transforms/Instrumentation
LLVM_LINT_FILTER=-,+whitespace
${CPPLINT} --filter=${LLVM_LINT_FILTER} ${LLVM_INSTRUMENTATION}/AddressSanitizer.cpp \
                                        ${LLVM_INSTRUMENTATION}/ThreadSanitizer.cpp \
                                        ${LLVM_INSTRUMENTATION}/BlackList.*

COMPILER_RT=projects/compiler-rt

# Headers
SANITIZER_INCLUDES=${COMPILER_RT}/include/sanitizer
${CPPLINT} --filter=${TSAN_RTL_LINT_FILTER} ${SANITIZER_INCLUDES}/*.h

# Sanitizer_common
COMMON_RTL=${COMPILER_RT}/lib/sanitizer_common
${CPPLINT} --filter=${ASAN_RTL_LINT_FILTER} ${COMMON_RTL}/*.{cc,h}
${CPPLINT} --filter=${TSAN_RTL_LINT_FILTER} ${COMMON_RTL}/tests/*.cc

#Interception
INTERCEPTION=${COMPILER_RT}/lib/interception
${CPPLINT} --filter=${ASAN_RTL_LINT_FILTER} ${INTERCEPTION}/*.{cc,h}

# ASan
ASAN_RTL=${COMPILER_RT}/lib/asan
${CPPLINT} --filter=${ASAN_RTL_LINT_FILTER} ${ASAN_RTL}/*.{cc,h}
${CPPLINT} --filter=${ASAN_TEST_LINT_FILTER} ${ASAN_RTL}/tests/*.{cc,h}
${CPPLINT} --filter=${ASAN_TEST_LINT_FILTER} ${ASAN_RTL}/lit_tests/*.cc \
                                             ${ASAN_RTL}/lit_tests/*/*.cc \

# TSan
TSAN_RTL=${COMPILER_RT}/lib/tsan
${CPPLINT} --filter=${TSAN_RTL_LINT_FILTER} ${TSAN_RTL}/rtl/*.{cc,h}
${CPPLINT} --filter=${TSAN_TEST_LINT_FILTER} ${TSAN_RTL}/tests/rtl/*.{cc,h} \
                                             ${TSAN_RTL}/tests/unit/*.cc \
                                             ${TSAN_RTL}/lit_tests/*.cc

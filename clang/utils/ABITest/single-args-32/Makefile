# Usage: make test.N.report 
#
# COUNT can be over-ridden to change the number of tests generated per
# file, and TESTARGS is used to change the type generation. Make sure
# to 'make clean' after changing either of these parameters.

COUNT := 1
TIMEOUT := 5

CFLAGS := -std=gnu99

X_COMPILER := gcc
X_LL_CFLAGS := -emit-llvm -S
Y_COMPILER := clang
Y_LL_CFLAGS := -emit-llvm -S
CC := gcc

X_CFLAGS := -m32
Y_CFLAGS := -m32
CC_CFLAGS := -m32

include ../Makefile.test.common

TESTARGS += --no-function-return --max-args 1

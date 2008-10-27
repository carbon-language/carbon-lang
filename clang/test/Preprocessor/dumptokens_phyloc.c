// RUN: clang -dump-tokens %s 2>&1 | grep "PhysLoc=.*dumptokens_phyloc.c:3:20"

#define TESTPHYLOC 10

TESTPHYLOC

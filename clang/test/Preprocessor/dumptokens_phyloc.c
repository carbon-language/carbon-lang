// RUN: clang -dumptokens %s 2>&1 | grep "PhysLoc=[_.a-zA-Z]*:3:20"

#define TESTPHYLOC 10

TESTPHYLOC

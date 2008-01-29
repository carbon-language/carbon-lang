// RUN: clang %s -emit-llvm

int testBoolAssign(void) {
int ss;
if ((ss = ss && ss)) {}
}

// RUN: clang %s -emit-llvm -o %t

int testBoolAssign(void) {
int ss;
if ((ss = ss && ss)) {}
}

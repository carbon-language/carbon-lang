// RUN: clang-cc %s -emit-llvm -o %t

int testBoolAssign(void) {
  int ss;
  if ((ss = ss && ss)) {}
  return 1;
}

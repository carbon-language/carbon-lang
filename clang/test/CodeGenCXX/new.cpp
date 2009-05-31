// RUN: clang-cc %s -emit-llvm -o %t

void t1() {
  int* a = new int;
}

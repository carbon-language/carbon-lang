// RUN: clang -emit-llvm -S -o %t1.ll %s &&
// RUN: clang -emit-ast -o %t.ast %s &&
// RUN: clang -emit-llvm -S -o %t2.ll %t.ast &&
// RUN: diff %t1.ll %t2.ll
// XFAIL: *

int main() {
  return 0;
}

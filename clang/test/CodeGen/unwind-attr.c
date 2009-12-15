// RUN: %clang_cc1 -fexceptions -emit-llvm -o - %s | grep "@foo()" | not grep nounwind
// RUN: %clang_cc1 -emit-llvm -o - %s | grep "@foo()" | grep nounwind 

int foo(void) {
  return 0;
}

// RUN: clang-cc -fexceptions -emit-llvm -o - %s | grep "@foo()" | not grep nounwind &&
// RUN: clang-cc -emit-llvm -o - %s | grep "@foo()" | grep nounwind 

int foo(void) {
  return 0;
}

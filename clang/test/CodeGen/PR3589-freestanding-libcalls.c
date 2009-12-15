// RUN: %clang_cc1 -emit-llvm %s -o - | grep 'declare i32 @printf' | count 1
// RUN: %clang_cc1 -O2 -emit-llvm %s -o - | grep 'declare i32 @puts' | count 1
// RUN: %clang_cc1 -ffreestanding -O2 -emit-llvm %s -o - | grep 'declare i32 @puts' | count 0

int printf(const char *, ...);

void f0() {
  printf("hello\n");
}

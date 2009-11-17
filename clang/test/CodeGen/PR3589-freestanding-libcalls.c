// RUN: clang-cc -emit-llvm %s -o - | grep 'declare i32 @printf' | count 1
// RUN: clang-cc -O2 -emit-llvm %s -o - | grep 'declare i32 @puts' | count 1
// RUN: clang-cc -ffreestanding -O2 -emit-llvm %s -o - | grep 'declare i32 @puts' | count 0

int printf(const char *, ...);

void f0() {
  printf("hello\n");
}

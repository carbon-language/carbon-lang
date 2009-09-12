// RUN: clang-cc -triple armv7-apple-darwin9 -emit-llvm -o - %s | FileCheck %s

// CHECK: define arm_apcscc signext i8 @f0()
char f0(void) {
  return 0;
}

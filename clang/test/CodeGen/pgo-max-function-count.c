// RUN: %clang -fprofile-generate -o %t -O2 %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw  %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: %clang -fprofile-use=%t.profdata -o - -S -emit-llvm %s | FileCheck %s
// Check
int main() {
  return 0;
}
// CHECK: !{{[0-9]+}} = !{i32 1, !"MaxFunctionCount", i32 1}

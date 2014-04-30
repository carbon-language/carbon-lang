// RUN: %clang_profgen -o %t -O3 %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: %clang_profuse=%t.profdata -o - -S -emit-llvm %s | FileCheck %s

int main(int argc, const char *argv[]) {
  // CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof !1
  if (argc)
    return 0;
  return 1;
}
// CHECK: !1 = metadata !{metadata !"branch_weights", i32 2, i32 1}

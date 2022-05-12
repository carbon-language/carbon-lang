// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s

// <rdar://problem/10465114>
struct overwrite_string_struct {
  char L[3];
  int M;
} overwrite_string[] = { { { @encode(void**) }, 1 }, [0].L[1] = 'x'};
// CHECK: [3 x i8] c"^xv", i32 1

// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name tu1.cpp %s | FileCheck %s

#include "Inputs/header1.h"

int main() {
  func(1);
  static_func(2);
}

// CHECK: static_func
// CHECK-NEXT: File 0, 12:32 -> 20:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 14:15 -> 16:4 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 16:10 -> 18:4 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 1, 3:10 -> 3:28 = #0 (HasCodeBefore = 0, Expanded file = 0)

// CHECK-NEXT: func
// CHECK-NEXT: File 0, 4:25 -> 11:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 6:15 -> 8:4 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 8:10 -> 10:4 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 1, 3:10 -> 3:28 = #0 (HasCodeBefore = 0, Expanded file = 0)

// CHECK-NEXT: static_func2
// CHECK-NEXT: File 0, 21:33 -> 29:2 = 0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 1, 3:10 -> 3:28 = 0 (HasCodeBefore = 0, Expanded file = 0)

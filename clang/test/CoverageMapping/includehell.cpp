// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name includehell.cpp %s | FileCheck %s

// CHECK: File 0, 1:1 -> 9:7 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 2:13 -> 4:2 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 4:8 -> 6:2 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 7:11 -> 9:2 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 9:8 -> 11:2 = (#0 - #2) (HasCodeBefore = 0)
int main() {               // CHECK-NEXT: File 1, [[@LINE]]:12 -> [[@LINE+4]]:2 = #0 (HasCodeBefore = 0)
  int x = 0;
  #include "Inputs/code.h" // CHECK-NEXT: Expansion,File 1, [[@LINE]]:12 -> [[@LINE]]:27 = #0 (HasCodeBefore = 0, Expanded file = 0)
  return 0;
}

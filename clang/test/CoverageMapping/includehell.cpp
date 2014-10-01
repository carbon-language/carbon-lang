// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name includehell.cpp %s | FileCheck %s

int main() {               // CHECK: File 0, [[@LINE]]:12 -> [[@LINE+4]]:2 = #0 (HasCodeBefore = 0)
  int x = 0;
  #include "Inputs/code.h" // CHECK-NEXT: Expansion,File 0, [[@LINE]]:12 -> [[@LINE]]:27 = #0 (HasCodeBefore = 0, Expanded file = 1)
  return 0;
}
// CHECK-NEXT: File 1, 1:1 -> 9:7 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 2:13 -> 4:2 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 4:8 -> 6:2 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 7:11 -> 9:2 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 9:8 -> 11:2 = (#0 - #2) (HasCodeBefore = 0)

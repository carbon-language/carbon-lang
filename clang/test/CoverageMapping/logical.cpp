// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name logical.cpp %s | FileCheck %s

int main() {                        // CHECK: File 0, [[@LINE]]:12 -> [[@LINE+10]]:2 = #0 (HasCodeBefore = 0)
  bool bt = true;
  bool bf = false;
  bool a = bt && bf;                // CHECK-NEXT: File 0, [[@LINE]]:18 -> [[@LINE]]:20 = #1 (HasCodeBefore = 0)
  a = bt &&
      bf;                           // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:9 = #2 (HasCodeBefore = 0)
  a = bf || bt;                     // CHECK-NEXT: File 0, [[@LINE]]:13 -> [[@LINE]]:15 = #3 (HasCodeBefore = 0)
  a = bf ||
      bt;                           // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:9 = #4 (HasCodeBefore = 0)
  return 0;
}

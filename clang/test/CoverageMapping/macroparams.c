// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macroparams.c %s | FileCheck %s

#define MACRO2(X2) (X2 + 2) // CHECK: File 0, [[@LINE]]:20 -> [[@LINE]]:28 = #0 (HasCodeBefore = 0)
#define MACRO(X) MACRO2(x)  // CHECK-NEXT: Expansion,File 1, [[@LINE]]:18 -> [[@LINE]]:24 = #0 (HasCodeBefore = 0, Expanded file = 0)
                            // CHECK-NEXT: File 1, [[@LINE-1]]:25 -> [[@LINE-1]]:26 = #0 (HasCodeBefore = 0)

int main() {                // CHECK-NEXT: File 2, [[@LINE]]:12 -> [[@LINE+4]]:2 = #0 (HasCodeBefore = 0)
  int x = 0;
  MACRO(x);                 // CHECK-NEXT: Expansion,File 2, [[@LINE]]:3 -> [[@LINE]]:8 = #0 (HasCodeBefore = 0, Expanded file = 1)
  return 0;
}

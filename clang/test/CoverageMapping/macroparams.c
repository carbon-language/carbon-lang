// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macroparams.c %s | FileCheck %s

#define MACRO2(X2) (X2 + 2) // CHECK-DAG: File 2, [[@LINE]]:20 -> [[@LINE]]:28 = #0 (HasCodeBefore = 0)
#define MACRO(X) MACRO2(x)  // CHECK-DAG: File 1, [[@LINE]]:25 -> [[@LINE]]:26 = #0 (HasCodeBefore = 0)
                            // CHECK-DAG: Expansion,File 1, [[@LINE-1]]:18 -> [[@LINE-1]]:24 = #0 (HasCodeBefore = 0, Expanded file = 2)


int main() {                // CHECK-DAG: File 0, [[@LINE]]:12 -> [[@LINE+4]]:2 = #0 (HasCodeBefore = 0)
  int x = 0;
  MACRO(x);                 // CHECK-DAG: Expansion,File 0, [[@LINE]]:3 -> [[@LINE]]:8 = #0 (HasCodeBefore = 0, Expanded file = 1)
  return 0;
}

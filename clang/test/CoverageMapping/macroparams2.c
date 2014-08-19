// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macroparams2.c %s | FileCheck %s

// A test case for when the first macro parameter is used after the second
// macro parameter.

struct S {
  int i, j;
};

#define MACRO(REFS, CALLS)  (4 * (CALLS) < (REFS))

int main() {                       // CHECK: File 0, [[@LINE]]:12 -> [[@LINE+7]]:2 = #0 (HasCodeBefore = 0)
  struct S arr[32] = { 0 };        // CHECK-NEXT: Expansion,File 0, [[@LINE+2]]:7 -> [[@LINE+2]]:12 = #0 (HasCodeBefore = 0, Expanded file = 1)
  int n = 0;                       // CHECK-NEXT: File 0, [[@LINE+1]]:13 -> [[@LINE+1]]:21 = #0 (HasCodeBefore = 0)
  if (MACRO(arr[n].j, arr[n].i)) { // CHECK-NEXT: File 0, [[@LINE]]:23 -> [[@LINE]]:31 = #0 (HasCodeBefore = 0)
    n = 1;                         // CHECK-NEXT: File 0, [[@LINE-1]]:34 -> [[@LINE+1]]:4 = #1 (HasCodeBefore = 0)
  }
  return n;
}                                  // CHECK-NEXT: File 1, [[@LINE-9]]:29 -> [[@LINE-9]]:51 = #0 (HasCodeBefore = 0


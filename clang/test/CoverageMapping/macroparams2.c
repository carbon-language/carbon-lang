// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macroparams2.c %s | FileCheck %s

// A test case for when the first macro parameter is used after the second
// macro parameter.

struct S {
  int i, j;
};

#define MACRO(REFS, CALLS)  (4 * (CALLS) < (REFS))

int main() {
  struct S arr[32] = { 0 };
  int n = 0;
  if (MACRO(arr[n].j, arr[n].i)) {
    n = 1;
  }
  return n;
}

// CHECK: File 0, 12:12 -> 19:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 0, 15:7 -> 15:12 = #0 (HasCodeBefore = 0, Expanded file = 1)
// CHECK-NEXT: File 0, 15:13 -> 15:21 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 15:23 -> 15:31 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 15:34 -> 17:4 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 10:29 -> 10:51 = #0 (HasCodeBefore = 0

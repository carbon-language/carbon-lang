// RUN: llvm-cov report %S/Inputs/reportFunctionPlacement.covmapping -instr-profile %S/Inputs/reportFunctionPlacement.profdata -no-colors 2>&1 | FileCheck %s
// This test checks that the functions defined in header files will get
// associated with header files rather than source files for the reports.

#include "Inputs/reportFunctionPlacement.h"

// CHECK: Filename                    Regions    Miss   Cover Functions  Executed
// CHECK: ---
// CHECK: ...ortFunctionPlacement.h         2       1  50.00%         2    50.00%
// CHECK: ...tFunctionPlacement.cpp         2       0 100.00%         2   100.00%
// CHECK: ---
// CHECK: TOTAL                             4       1  75.00%         4    75.00%

void func() {
}

int main() {
  foo(10);
  func();
  return 0;
}

// llvm-cov doesn't work on big endian yet
// XFAIL: powerpc64-, s390x, mips-, mips64-, sparc

// RUN: not %clang_cc1 -triple i686-pc-win32 -fexceptions -fms-extensions -emit-llvm -o - %s 2>&1 | FileCheck %s

// This is a codegen test because we only emit the diagnostic when we start
// generating code.

int SaveDiv(int numerator, int denominator, int *res) {
  int myres = 0;
  __try {
    myres = numerator / denominator;
  } __except (1) {
    return 0;
  }
  *res = myres;
  return 1;
}
// CHECK-NOT error
// CHECK: error: cannot compile this SEH __try yet
// CHECK-NOT error

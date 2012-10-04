// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// CHECK: @_Z9no_return
int no_return() {
  // CHECK: unreachable
}

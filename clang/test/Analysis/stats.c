// RUN: %clang_cc1 -analyze -analyzer-stats %s 2>&1 | FileCheck %s
// XFAIL: *

void foo() {
  ;
}
// CHECK: ... Statistics Collected ...

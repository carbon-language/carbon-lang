// RUN: %clang_cc1 -analyze -analyzer-stats %s 2> FileCheck

void foo() {
  ;
}
// CHECK: ... Statistics Collected ...

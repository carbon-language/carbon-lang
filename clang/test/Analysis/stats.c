// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-stats %s > FileCheck %s

void foo() {
  int x;
}
// CHECK: ... Statistics Collected ...
// CHECK:The # of times RemoveDeadBindings is called

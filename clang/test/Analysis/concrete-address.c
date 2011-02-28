// RUN: %clang_cc1 -analyze -analyzer-checker=core,core.experimental -analyzer-store=basic -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,core.experimental -analyzer-store=region -verify %s

void foo() {
  int *p = (int*) 0x10000; // Should not crash here.
  *p = 3;
}

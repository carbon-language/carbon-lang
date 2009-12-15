// RUN: %clang_cc1 -ftrapv %s -emit-llvm -o %t
// RUN: grep "__overflow_handler" %t | count 2

unsigned int ui, uj, uk;
int i, j, k;

void foo() {
  ui = uj + uk;
  i = j + k;
}

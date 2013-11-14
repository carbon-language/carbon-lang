// RUN: clang-tidy %s -checks='clang-analyzer-cplusplus' -- | FileCheck %s

void f() {
  int *p = new int(42);
  delete p;
  delete p;
  // CHECK: warning: Attempt to free released memory
}

// RUN: %clangxx -fsanitize=function %s -O3 -g -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdint.h>

void f() {}

void g(int x) {}

int main(void) {
  // CHECK: runtime error: call to function f() through pointer to incorrect function type 'void (*)(int)'
  // CHECK-NEXT: function.cpp:6: note: f() defined here
  reinterpret_cast<void (*)(int)>(reinterpret_cast<uintptr_t>(f))(42);

  // CHECK-NOT: runtime error: call to function g
  reinterpret_cast<void (*)(int)>(reinterpret_cast<uintptr_t>(g))(42);
}

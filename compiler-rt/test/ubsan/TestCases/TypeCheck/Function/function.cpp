// RUN: %clangxx -fsanitize=function %s -O3 -g -o %t
// RUN: %run %t 2>&1 | FileCheck %s
// Verify that we can disable symbolization if needed:
// RUN: UBSAN_OPTIONS=symbolize=0 ASAN_OPTIONS=symbolize=0 %run %t 2>&1 | FileCheck %s --check-prefix=NOSYM

// -fsanitize=function is unsupported on Darwin yet.
// XFAIL: darwin

#include <stdint.h>

void f() {}

void g(int x) {}

int main(void) {
  // CHECK: runtime error: call to function f() through pointer to incorrect function type 'void (*)(int)'
  // CHECK-NEXT: function.cpp:11: note: f() defined here
  // NOSYM: runtime error: call to function (unknown) through pointer to incorrect function type 'void (*)(int)'
  // NOSYM-NEXT: ({{.*}}+0x{{.*}}): note: (unknown) defined here
  reinterpret_cast<void (*)(int)>(reinterpret_cast<uintptr_t>(f))(42);

  // CHECK-NOT: runtime error: call to function g
  reinterpret_cast<void (*)(int)>(reinterpret_cast<uintptr_t>(g))(42);
}

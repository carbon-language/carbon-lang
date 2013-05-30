// RUN: %clang_cc1 -fms-compatibility -fsyntax-only -verify %s
// expected-no-diagnostics

#include <stddef.h>

struct arbitrary_t {} arbitrary;
void *operator new(size_t size, arbitrary_t);

void f() {
  // Expect no error in MSVC compatibility mode
  int *p = new(arbitrary) int[4];
}

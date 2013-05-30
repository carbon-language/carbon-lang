// RUN: %clang_cc1 -fms-compatibility -fsyntax-only -verify %s
// expected-no-diagnostics

struct arbitrary_t {} arbitrary;
void *operator new(unsigned int size, arbitrary_t);

void f() {
  // Expect no error in MSVC compatibility mode
  int *p = new(arbitrary) int[4];
}

// Test __sanitizer_get_total_unique_coverage for caller-callee coverage

// RUN: %clangxx_asan -fsanitize-coverage=4 %s -o %t
// RUN: ASAN_OPTIONS=coverage=1 %run %t
// RUN: rm -f caller-callee*.sancov
//
// REQUIRES: asan-64-bits

#include <sanitizer/common_interface_defs.h>
#include <stdio.h>
#include <assert.h>
int P = 0;
struct Foo {virtual void f() {if (P) printf("Foo::f()\n");}};
struct Foo1 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};
struct Foo2 : Foo {virtual void f() {if (P) printf("%d\n", __LINE__);}};

Foo *foo[3] = {new Foo, new Foo1, new Foo2};

uintptr_t CheckNewTotalUniqueCoverageIsLargerAndReturnIt(uintptr_t old_total) {
  uintptr_t new_total = __sanitizer_get_total_unique_coverage();
  assert(new_total > old_total);
  return new_total;
}

int main(int argc, char **argv) {
  uintptr_t total = CheckNewTotalUniqueCoverageIsLargerAndReturnIt(0);
  foo[0]->f();
  total = CheckNewTotalUniqueCoverageIsLargerAndReturnIt(total);
  foo[1]->f();
  total = CheckNewTotalUniqueCoverageIsLargerAndReturnIt(total);
  foo[2]->f();
  total = CheckNewTotalUniqueCoverageIsLargerAndReturnIt(total);
  // Ok, called every function once.
  // Now call them again from another call site. Should get new coverage.
  foo[0]->f();
  total = CheckNewTotalUniqueCoverageIsLargerAndReturnIt(total);
  foo[1]->f();
  total = CheckNewTotalUniqueCoverageIsLargerAndReturnIt(total);
  foo[2]->f();
  total = CheckNewTotalUniqueCoverageIsLargerAndReturnIt(total);
}

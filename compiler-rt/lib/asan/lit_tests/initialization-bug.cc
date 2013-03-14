// Test to make sure basic initialization order errors are caught.

// RUN: %clangxx_asan -m64 -O0 %s %p/Helpers/initialization-bug-extra2.cc -o %t
// RUN: ASAN_OPTIONS=check_initialization_order=true %t 2>&1 \
// RUN:    | %symbolize | FileCheck %s
// RUN: %clangxx_asan -m32 -O0 %s %p/Helpers/initialization-bug-extra2.cc -o %t
// RUN: ASAN_OPTIONS=check_initialization_order=true %t 2>&1 \
// RUN:     | %symbolize | FileCheck %s

// Do not test with optimization -- the error may be optimized away.

#include <cstdio>

// The structure of the test is:
// "x", "y", "z" are dynamically initialized globals.
// Value of "x" depends on "y", value of "y" depends on "z".
// "x" and "z" are defined in this TU, "y" is defined in another one.
// Thus we shoud stably report initialization order fiasco independently of
// the translation unit order.

int initZ() {
  return 5;
}
int z = initZ();

// 'y' is a dynamically initialized global residing in a different TU.  This
// dynamic initializer will read the value of 'y' before main starts.  The
// result is undefined behavior, which should be caught by initialization order
// checking.
extern int y;
int __attribute__((noinline)) initX() {
  return y + 1;
  // CHECK: {{AddressSanitizer: initialization-order-fiasco}}
  // CHECK: {{READ of size .* at 0x.* thread T0}}
  // CHECK: {{0x.* is located 0 bytes inside of global variable .*(y|z).*}}
}

// This initializer begins our initialization order problems.
static int x = initX();

int main() {
  // ASan should have caused an exit before main runs.
  printf("PASS\n");
  // CHECK-NOT: PASS
  return 0;
}

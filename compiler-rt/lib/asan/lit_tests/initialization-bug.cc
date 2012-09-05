// Test to make sure basic initialization order errors are caught.

// RUN: %clangxx_asan -m64 -O0 %s %p/Helpers/initialization-bug-extra.cc\
// RUN:   -mllvm -asan-initialization-order -o %t && %t 2>&1 \
// RUN:    | %symbolize | FileCheck %s
// RUN: %clangxx_asan -m32 -O0 %s %p/Helpers/initialization-bug-extra.cc\
// RUN:   -mllvm -asan-initialization-order -o %t && %t 2>&1 \
// RUN:     | %symbolize | FileCheck %s

// Do not test with optimization -- the error may be optimized away.

extern int y;

// 'y' is a dynamically initialized global residing in a different TU.  This
// dynamic initializer will read the value of 'y' before main starts.  The
// result is undefined behavior, which should be caught by initialization order
// checking.
int __attribute__((noinline)) initX() {
  return y + 1;
  // CHECK: {{AddressSanitizer initialization-order-fiasco}}
  // CHECK: {{READ of size .* at 0x.* thread T0}}
  // CHECK: {{#0 0x.* in .*initX.* .*initialization-bug.cc:19}}
  // CHECK: {{0x.* is located 0 bytes inside of global variable .*y.*}}
}


// This initializer begins our initialization order problems.
static int x = initX();

// ASan should have caused an exit before main runs.
int main() {
  return -1;
}

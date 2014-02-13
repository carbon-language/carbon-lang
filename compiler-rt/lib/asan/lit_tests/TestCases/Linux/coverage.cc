// RUN: %clangxx_asan -mllvm -asan-coverage=1 -DSHARED %s -shared -o %t.so -fPIC
// RUN: %clangxx_asan -mllvm -asan-coverage=1 %s   -o %t -Wl,-R. %t.so
// RUN: export ASAN_OPTIONS=coverage=1:verbosity=1
// RUN: %t 2>&1         | FileCheck %s --check-prefix=CHECK-main
// RUN: %t foo 2>&1     | FileCheck %s --check-prefix=CHECK-foo
// RUN: %t bar 2>&1     | FileCheck %s --check-prefix=CHECK-bar
// RUN: %t foo bar 2>&1 | FileCheck %s --check-prefix=CHECK-foo-bar
// RUN: not %t foo bar 1 2  2>&1 | FileCheck %s --check-prefix=CHECK-report
//
// https://code.google.com/p/address-sanitizer/issues/detail?id=263
// XFAIL: android

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#ifdef SHARED
void bar() { printf("bar\n"); }
#else
__attribute__((noinline))
void foo() { printf("foo\n"); }
extern void bar();

int G[4];

int main(int argc, char **argv) {
  fprintf(stderr, "PID: %d\n", getpid());
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "foo"))
      foo();
    if (!strcmp(argv[i], "bar"))
      bar();
  }
  return G[argc];  // Buffer overflow if argc >= 4.
}
#endif

// CHECK-main: PID: [[PID:[0-9]+]]
// CHECK-main: [[PID]].sancov: 1 PCs written
// CHECK-main-NOT: .so.[[PID]]
//
// CHECK-foo: PID: [[PID:[0-9]+]]
// CHECK-foo: [[PID]].sancov: 2 PCs written
// CHECK-foo-NOT: .so.[[PID]]
//
// CHECK-bar: PID: [[PID:[0-9]+]]
// CHECK-bar: [[PID]].sancov: 1 PCs written
// CHECK-bar: .so.[[PID]].sancov: 1 PCs written
//
// CHECK-foo-bar: PID: [[PID:[0-9]+]]
// CHECK-foo-bar: [[PID]].sancov: 2 PCs written
// CHECK-foo-bar: so.[[PID]].sancov: 1 PCs written
//
// CHECK-report: AddressSanitizer: global-buffer-overflow
// CHECK-report: PCs written

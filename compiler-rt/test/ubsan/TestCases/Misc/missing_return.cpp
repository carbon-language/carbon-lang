// RUN: %clangxx -fsanitize=return -g %s -O3 -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: UBSAN_OPTIONS=print_stacktrace=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os-STACKTRACE

// CHECK: missing_return.cpp:[[@LINE+1]]:5: runtime error: execution reached the end of a value-returning function without returning a value
int f() {
// Slow stack unwinding is disabled on Darwin for now, see
// https://code.google.com/p/address-sanitizer/issues/detail?id=137
// CHECK-Linux-STACKTRACE: #0 {{.*}} in f(){{.*}}missing_return.cpp:[[@LINE-3]]
// Check for already checked line to avoid lit error reports.
// CHECK-Darwin-STACKTRACE: missing_return.cpp
}

int main(int, char **argv) {
  return f();
}

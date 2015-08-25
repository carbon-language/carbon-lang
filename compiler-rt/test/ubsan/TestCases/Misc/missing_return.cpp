// RUN: %clangxx -fsanitize=return -g %s -O3 -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %env_ubsan_opts=print_stacktrace=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-%os-STACKTRACE

// CHECK: missing_return.cpp:[[@LINE+1]]:5: runtime error: execution reached the end of a value-returning function without returning a value
int f() {
// Slow stack unwinding is not available on Darwin for now, see
// https://code.google.com/p/address-sanitizer/issues/detail?id=137
// CHECK-Linux-STACKTRACE: #0 {{.*}}f(){{.*}}missing_return.cpp:[[@LINE-3]]
// CHECK-FreeBSD-STACKTRACE: #0 {{.*}}f(void){{.*}}missing_return.cpp:[[@LINE-4]]
}

int main(int, char **argv) {
  return f();
}

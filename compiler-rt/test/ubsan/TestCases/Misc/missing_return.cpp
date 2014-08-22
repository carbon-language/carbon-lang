// RUN: %clangxx -fsanitize=return -g %s -O3 -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: UBSAN_OPTIONS=print_stacktrace=1 not %run %t 2>&1 | FileCheck %s --check-prefix=STACKTRACE

// CHECK: missing_return.cpp:[[@LINE+1]]:5: runtime error: execution reached the end of a value-returning function without returning a value
int f() {
// STACKTRACE: #0 {{.*}} in f(){{.*}}missing_return.cpp:[[@LINE-1]]
}

int main(int, char **argv) {
  return f();
// STACKTRACE: #1 {{.*}} in main{{.*}}missing_return.cpp:[[@LINE-1]]
}

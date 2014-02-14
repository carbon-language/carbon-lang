// RUN: %clangxx -fsanitize=return %s -O3 -o %t && %t 2>&1 | FileCheck %s

// CHECK: missing_return.cpp:4:5: runtime error: execution reached the end of a value-returning function without returning a value
int f() {
}

int main(int, char **argv) {
  return f();
}

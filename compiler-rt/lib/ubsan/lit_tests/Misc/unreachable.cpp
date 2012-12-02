// RUN: %clang -fsanitize=unreachable %s -O3 -o %t && %t 2>&1 | FileCheck %s

int main(int, char **argv) {
  // CHECK: unreachable.cpp:5:3: runtime error: execution reached a __builtin_unreachable() call
  __builtin_unreachable();
}

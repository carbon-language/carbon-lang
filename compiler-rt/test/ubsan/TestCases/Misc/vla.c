// RUN: %clang -fsanitize=vla-bound %s -O3 -o %t
// RUN: %t 2>&1 | FileCheck %s --check-prefix=CHECK-MINUS-ONE
// RUN: %t a 2>&1 | FileCheck %s --check-prefix=CHECK-ZERO
// RUN: %t a b

int main(int argc, char **argv) {
  // CHECK-MINUS-ONE: vla.c:9:11: runtime error: variable length array bound evaluates to non-positive value -1
  // CHECK-ZERO: vla.c:9:11: runtime error: variable length array bound evaluates to non-positive value 0
  int arr[argc - 2];
  return 0;
}

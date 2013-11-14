// RUN: %clangxx -fsanitize=bounds %s -O3 -o %T/bounds.exe
// RUN: %T/bounds.exe 0 0 0
// RUN: %T/bounds.exe 1 2 3
// RUN: %T/bounds.exe 2 0 0 2>&1 | FileCheck %s --check-prefix=CHECK-A-2
// RUN: %T/bounds.exe 0 3 0 2>&1 | FileCheck %s --check-prefix=CHECK-B-3
// RUN: %T/bounds.exe 0 0 4 2>&1 | FileCheck %s --check-prefix=CHECK-C-4

int main(int argc, char **argv) {
  int arr[2][3][4] = {};

  return arr[argv[1][0] - '0'][argv[2][0] - '0'][argv[3][0] - '0'];
  // CHECK-A-2: bounds.cpp:11:10: runtime error: index 2 out of bounds for type 'int [2][3][4]'
  // CHECK-B-3: bounds.cpp:11:10: runtime error: index 3 out of bounds for type 'int [3][4]'
  // CHECK-C-4: bounds.cpp:11:10: runtime error: index 4 out of bounds for type 'int [4]'
}

// RUN: %clang_cl_asan -O0 %s -Fe%t
// RUN: echo "42" | %run %t 2>&1 | FileCheck %s

#include <iostream>

int main() {
  int i;
  std::cout << "Type i: ";
  std::cin >> i;
  return 0;
// CHECK: Type i:
// CHECK-NOT: AddressSanitizer: stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
}

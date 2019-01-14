// RUN: %clang -fsanitize=alignment %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK

int main(int argc, char* argv[]) {
// CHECK-NOT: alignment-assumption

  __builtin_assume_aligned(argv, 0x80000000);
// CHECK: alignment-assumption
// CHECK-NOT: alignment-assumption

  return 0;
}

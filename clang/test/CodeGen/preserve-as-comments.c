// RUN: %clang_cc1 -S -fno-preserve-as-comments %s -o - | FileCheck %s --check-prefix=NOASM --check-prefix=CHECK
// RUN: %clang_cc1 -S %s -o - | FileCheck %s --check-prefix=ASM --check-prefix=CHECK

// CHECK-LABEL: main
// CHECK: #APP
// ASM: #comment
// NOASM-NOT: #comment
// CHECK: #NO_APP
int main() {
  __asm__("/*comment*/");
  return 0;
}

// RUN: %clang_cc1 -S -triple=x86_64-unknown-unknown -fno-preserve-as-comments %s -o - | FileCheck %s --check-prefix=NOASM --check-prefix=CHECK
// RUN: %clang_cc1 -S %s -triple=x86_64-unknown-unknown -o - | FileCheck %s --check-prefix=ASM --check-prefix=CHECK

// CHECK-LABEL: main
// CHECK: #APP
// ASM: #comment
// NOASM-NOT: #comment
// CHECK: #NO_APP
int main() {
  __asm__("/*comment*/");
  return 0;
}

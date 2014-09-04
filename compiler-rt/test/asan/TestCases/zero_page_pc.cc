// Check that ASan correctly detects SEGV on the zero page.
// RUN: %clangxx_asan %s -o %t && not %run %t 2>&1 | FileCheck %s

typedef void void_f();
int main() {
  void_f *func = (void_f *)0x7;
  func();
  // CHECK: {{AddressSanitizer: SEGV.*(pc.*0007)}}
  return 0;
}

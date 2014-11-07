// Check that ASan correctly detects SEGV on the zero page.
// RUN: %clangxx_asan %s -o %t && not %run %t 2>&1 | FileCheck %s

// https://code.google.com/p/address-sanitizer/issues/detail?id=336
// XFAIL: android

typedef void void_f();
int main() {
  void_f *func = (void_f *)0x4;
  func();
  // CHECK: {{AddressSanitizer: SEGV.*(pc.*0004)}}
  return 0;
}

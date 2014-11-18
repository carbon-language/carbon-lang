// Check that ASan correctly detects SEGV on the zero page.
// RUN: %clangxx_asan %s -o %t && not %run %t 2>&1 | FileCheck %s

typedef void void_f();
int main() {
  void_f *func = (void_f *)0x4;
  func();
  // x86 reports the SEGV with both address=4 and pc=4.
  // PowerPC64 reports it with address=4 but pc still in main().
  // CHECK: {{AddressSanitizer: SEGV.*(address|pc) 0x0*4}}
  return 0;
}

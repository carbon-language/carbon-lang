// Test that asan_symbolize does not hang when provided with an non-existing
// path.
// RUN: echo '#0 0xabcdabcd (%t/bad/path+0x1234)' | %asan_symbolize | FileCheck %s -check-prefix CHECK-BAD-FILE
// Note: can't check for "0xabcdabcd in ?? ??:0" since DarwinSymbolizer will print the file even if it doesn't exist.
// CHECK-BAD-FILE: #0 0xabcdabcd
// CHECK-BAD-FILE-EMPTY:

// Also test that asan_symbolize doesn't assert on an invalid address with a valid file:
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: echo '#0 0xabcdabcd (%t+0xabcdabcd)' | %asan_symbolize | FileCheck %s -check-prefix CHECK-BAD-ADDR
// CHECK-BAD-ADDR: #0 0xabcdabcd
// CHECK-BAD-ADDR-EMPTY:

int main() {
  return 0;
}

// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android
//
// Test that kernel area is not sanitized on 32-bit machines.
//
// RUN: %clangxx_asan %s -o %t
// RUN: ASAN_OPTIONS=verbosity=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%kernel_bits
// RUN: ASAN_OPTIONS=verbosity=1:full_address_space=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%kernel_bits
// RUN: ASAN_OPTIONS=verbosity=1:full_address_space=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-kernel-64-bits
//
// CHECK-kernel-32-bits: || `[0x38000000, 0xbfffffff]` || HighMem    ||
// CHECK-kernel-32-bits: || `[0x27000000, 0x37ffffff]` || HighShadow ||
// CHECK-kernel-32-bits: || `[0x24000000, 0x26ffffff]` || ShadowGap  ||
//
// CHECK-kernel-64-bits: || `[0x40000000, 0xffffffff]` || HighMem    ||
// CHECK-kernel-64-bits: || `[0x28000000, 0x3fffffff]` || HighShadow ||
// CHECK-kernel-64-bits: || `[0x24000000, 0x27ffffff]` || ShadowGap  ||
//
// REQUIRES: asan-32-bits

int main() {
  return 0;
}


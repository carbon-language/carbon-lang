// RUN: %clang_cc1 -O3 -triple arm64-apple-ios7 -target-feature +neon -S -ffreestanding %s -o - -target-cpu cyclone | FileCheck %s
// REQUIRES: arm64-registered-target
// test code generation for <rdar://problem/11487757>
#include <arm_neon.h>

unsigned bar();

// Branch if any lane of V0 is zero; 64 bit => !min
unsigned anyZero64(uint16x4_t a) {
// CHECK: anyZero64:
// CHECK: uminv.8b b[[REGNO1:[0-9]+]], v0
// CHECK-NEXT: fmov w[[REGNO2:[0-9]+]], s[[REGNO1]]
// CHECK-NEXT: cbz w[[REGNO2]], [[LABEL:[.A-Z_0-9]+]]
// CHECK: [[LABEL]]:
// CHECK-NEXT: b {{_bar|bar}}
  if (!vminv_u8(a))
    return bar();
  return 0;
}

// Branch if any lane of V0 is zero; 128 bit => !min
unsigned anyZero128(uint16x8_t a) {
// CHECK: anyZero128:
// CHECK: uminv.16b b[[REGNO1:[0-9]+]], v0
// CHECK-NEXT: fmov w[[REGNO2:[0-9]+]], s[[REGNO1]]
// CHECK-NEXT: cbz w[[REGNO2]], [[LABEL:[.A-Z_0-9]+]]
// CHECK: [[LABEL]]:
// CHECK-NEXT: b {{_bar|bar}}
  if (!vminvq_u8(a))
    return bar();
  return 0;
}

// Branch if any lane of V0 is non-zero; 64 bit => max
unsigned anyNonZero64(uint16x4_t a) {
// CHECK: anyNonZero64:
// CHECK: umaxv.8b b[[REGNO1:[0-9]+]], v0
// CHECK-NEXT: fmov w[[REGNO2:[0-9]+]], s[[REGNO1]]
// CHECK-NEXT: cbz w[[REGNO2]], [[LABEL:[.A-Z_0-9]+]]
// CHECK: [[LABEL]]:
// CHECK-NEXT: movz w0, #0
  if (vmaxv_u8(a))
    return bar();
  return 0;
}

// Branch if any lane of V0 is non-zero; 128 bit => max
unsigned anyNonZero128(uint16x8_t a) {
// CHECK: anyNonZero128:
// CHECK: umaxv.16b b[[REGNO1:[0-9]+]], v0
// CHECK-NEXT: fmov w[[REGNO2:[0-9]+]], s[[REGNO1]]
// CHECK-NEXT: cbz w[[REGNO2]], [[LABEL:[.A-Z_0-9]+]]
// CHECK: [[LABEL]]:
// CHECK-NEXT: movz w0, #0
  if (vmaxvq_u8(a))
    return bar();
  return 0;
}

// Branch if all lanes of V0 are zero; 64 bit => !max
unsigned allZero64(uint16x4_t a) {
// CHECK: allZero64:
// CHECK: umaxv.8b b[[REGNO1:[0-9]+]], v0
// CHECK-NEXT: fmov w[[REGNO2:[0-9]+]], s[[REGNO1]]
// CHECK-NEXT: cbz w[[REGNO2]], [[LABEL:[.A-Z_0-9]+]]
// CHECK: [[LABEL]]:
// CHECK-NEXT: b {{_bar|bar}}
  if (!vmaxv_u8(a))
    return bar();
  return 0;
}

// Branch if all lanes of V0 are zero; 128 bit => !max
unsigned allZero128(uint16x8_t a) {
// CHECK: allZero128:
// CHECK: umaxv.16b b[[REGNO1:[0-9]+]], v0
// CHECK-NEXT: fmov w[[REGNO2:[0-9]+]], s[[REGNO1]]
// CHECK-NEXT: cbz w[[REGNO2]], [[LABEL:[.A-Z_0-9]+]]
// CHECK: [[LABEL]]:
// CHECK-NEXT: b {{_bar|bar}}
  if (!vmaxvq_u8(a))
    return bar();
  return 0;
}

// Branch if all lanes of V0 are non-zero; 64 bit => min
unsigned allNonZero64(uint16x4_t a) {
// CHECK: allNonZero64:
// CHECK: uminv.8b b[[REGNO1:[0-9]+]], v0
// CHECK-NEXT: fmov w[[REGNO2:[0-9]+]], s[[REGNO1]]
// CHECK-NEXT: cbz w[[REGNO2]], [[LABEL:[.A-Z_0-9]+]]
// CHECK: [[LABEL]]:
// CHECK-NEXT: movz w0, #0
  if (vminv_u8(a))
    return bar();
  return 0;
}

// Branch if all lanes of V0 are non-zero; 128 bit => min
unsigned allNonZero128(uint16x8_t a) {
// CHECK: allNonZero128:
// CHECK: uminv.16b b[[REGNO1:[0-9]+]], v0
// CHECK-NEXT: fmov w[[REGNO2:[0-9]+]], s[[REGNO1]]
// CHECK-NEXT: cbz w[[REGNO2]], [[LABEL:[.A-Z_0-9]+]]
// CHECK: [[LABEL]]:
// CHECK-NEXT: movz w0, #0
  if (vminvq_u8(a))
    return bar();
  return 0;
}


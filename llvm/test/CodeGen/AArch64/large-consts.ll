; RUN: llc -mtriple=arm64-linux-gnu -o - %s -code-model=large -show-mc-encoding | FileCheck %s --check-prefix=CHECK-ARM64

; Make sure the shift amount is encoded into the instructions by LLVM because
; it's not the linker's job to put it there.

define double @foo() {

; CHECK-ARM64: movz [[CPADDR:x[0-9]+]], #:abs_g3:.LCPI0_0   // encoding: [0bAAA01000,A,0b111AAAAA,0xd2]
; CHECK-ARM64: movk [[CPADDR]], #:abs_g2_nc:.LCPI0_0 // encoding: [0bAAA01000,A,0b110AAAAA,0xf2]
; CHECK-ARM64: movk [[CPADDR]], #:abs_g1_nc:.LCPI0_0 // encoding: [0bAAA01000,A,0b101AAAAA,0xf2]
; CHECK-ARM64: movk [[CPADDR]], #:abs_g0_nc:.LCPI0_0 // encoding: [0bAAA01000,A,0b100AAAAA,0xf2]

  ret double 3.14159
}

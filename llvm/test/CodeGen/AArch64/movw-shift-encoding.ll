; RUN: llc -mtriple=aarch64-linux-gnu < %s -show-mc-encoding -code-model=large | FileCheck %s --check-prefix=CHECK-AARCH64
; RUN: llc -mtriple=arm64-linux-gnu < %s -show-mc-encoding -code-model=large | FileCheck %s --check-prefix=CHECK-ARM64

@var = global i32 0

; CodeGen should ensure that the correct shift bits are set, because the linker
; isn't going to!

define i32* @get_var() {
  ret i32* @var
; CHECK-AARCH64: movz    x0, #:abs_g3:var        // encoding: [A,A,0xe0'A',0xd2'A']
; CHECK-AARCH64: movk    x0, #:abs_g2_nc:var     // encoding: [A,A,0xc0'A',0xf2'A']
; CHECK-AARCH64: movk    x0, #:abs_g1_nc:var     // encoding: [A,A,0xa0'A',0xf2'A']
; CHECK-AARCH64: movk    x0, #:abs_g0_nc:var     // encoding: [A,A,0x80'A',0xf2'A']

; CHECK-ARM64: movz    x0, #:abs_g3:var        // encoding: [0bAAA00000,A,0b111AAAAA,0xd2]
; CHECK-ARM64: movk    x0, #:abs_g2_nc:var     // encoding: [0bAAA00000,A,0b110AAAAA,0xf2]
; CHECK-ARM64: movk    x0, #:abs_g1_nc:var     // encoding: [0bAAA00000,A,0b101AAAAA,0xf2]
; CHECK-ARM64: movk    x0, #:abs_g0_nc:var     // encoding: [0bAAA00000,A,0b100AAAAA,0xf2]
}

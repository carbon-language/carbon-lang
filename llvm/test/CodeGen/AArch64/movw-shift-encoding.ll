; RUN: llc -mtriple=aarch64-linux-gnu < %s -show-mc-encoding -code-model=large | FileCheck %s

@var = global i32 0

; CodeGen should ensure that the correct shift bits are set, because the linker
; isn't going to!

define i32* @get_var() {
  ret i32* @var
; CHECK: movz    x0, #:abs_g3:var        // encoding: [A,A,0xe0'A',0xd2'A']
; CHECK: movk    x0, #:abs_g2_nc:var     // encoding: [A,A,0xc0'A',0xf2'A']
; CHECK: movk    x0, #:abs_g1_nc:var     // encoding: [A,A,0xa0'A',0xf2'A']
; CHECK: movk    x0, #:abs_g0_nc:var     // encoding: [A,A,0x80'A',0xf2'A']
}

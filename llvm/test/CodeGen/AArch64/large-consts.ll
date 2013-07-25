; RUN: llc -mtriple=aarch64-linux-gnu -o - %s -code-model=large -show-mc-encoding | FileCheck %s

; Make sure the shift amount is encoded into the instructions by LLVM because
; it's not the linker's job to put it there.

define double @foo() {
; CHECK: movz [[CPADDR:x[0-9]+]], #:abs_g3:.LCPI0_0   // encoding: [A,A,0xe0'A',0xd2'A']
; CHECK: movk [[CPADDR]], #:abs_g2_nc:.LCPI0_0 // encoding: [A,A,0xc0'A',0xf2'A']
; CHECK: movk [[CPADDR]], #:abs_g1_nc:.LCPI0_0 // encoding: [A,A,0xa0'A',0xf2'A']
; CHECK: movk [[CPADDR]], #:abs_g0_nc:.LCPI0_0 // encoding: [A,A,0x80'A',0xf2'A']

  ret double 3.14159
}

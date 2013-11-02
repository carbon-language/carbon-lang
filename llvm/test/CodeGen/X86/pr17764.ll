; RUN: llc < %s -mtriple=x86_64-linux -mcpu=core-avx2 | FileCheck %s

define <16 x i16> @foo(<16 x i1> %mask, <16 x i16> %x, <16 x i16> %y) {
  %ret = select <16 x i1> %mask, <16 x i16> %x, <16 x i16> %y
  ret <16 x i16> %ret
}

; CHECK: foo
; CHECK: vpblendvb %ymm0, %ymm1, %ymm2, %ymm0
; CHECK: ret

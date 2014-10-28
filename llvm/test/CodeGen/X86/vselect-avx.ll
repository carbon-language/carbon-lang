; RUN: llc %s -o - -mattr=+avx | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

; For this test we used to optimize the <i1 true, i1 false, i1 false, i1 true>
; mask into <i32 2147483648, i32 0, i32 0, i32 2147483648> because we thought
; we would lower that into a blend where only the high bit is relevant.
; However, since the whole mask is constant, this is simplified incorrectly
; by the generic code, because it was expecting -1 in place of 2147483648.
; 
; The problem does not occur without AVX, because vselect of v4i32 is not legal
; nor custom.
;
; <rdar://problem/18675020>

; CHECK-LABEL: test:
; CHECK: vmovdqa {{.*#+}} xmm0 = [65535,0,0,65535]
; CHECK: vmovdqa {{.*#+}} xmm2 = [65533,124,125,14807]
; CHECK: ret
define void @test(<4 x i16>* %a, <4 x i16>* %b) {
body:
  %predphi = select <4 x i1> <i1 true, i1 false, i1 false, i1 true>, <4 x i16> <i16 -3, i16 545, i16 4385, i16 14807>, <4 x i16> <i16 123, i16 124, i16 125, i16 127>
  %predphi42 = select <4 x i1> <i1 true, i1 false, i1 false, i1 true>, <4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>, <4 x i16> zeroinitializer
  store <4 x i16> %predphi, <4 x i16>* %a, align 8
  store <4 x i16> %predphi42, <4 x i16>* %b, align 8
  ret void
}

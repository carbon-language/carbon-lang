; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

define <4 x i32> @trunc_64_32(<4 x i64> %A) nounwind uwtable readnone ssp{
; CHECK: trunc_64_32
; CHECK: pshufd
  %B = trunc <4 x i64> %A to <4 x i32>
  ret <4 x i32>%B
}
define <8 x i16> @trunc_32_16(<8 x i32> %A) nounwind uwtable readnone ssp{
; CHECK: trunc_32_16
; CHECK: pshufb
  %B = trunc <8 x i32> %A to <8 x i16>
  ret <8 x i16>%B
}

define <8 x i16> @trunc_after_setcc(<8 x float> %a, <8 x float> %b, <8 x float> %c, <8 x float> %d) {
; CHECK: trunc_after_setcc
; CHECK: vcmpltps
; CHECK-NOT: vextract
; CHECK: vcmpltps
; CHECK-NEXT: vandps
; CHECK-NEXT: vandps
; CHECK: ret
  %res1 = fcmp olt <8 x float> %a, %b
  %res2 = fcmp olt <8 x float> %c, %d
  %andr = and <8 x i1>%res1, %res2
  %ex = zext <8 x i1> %andr to <8 x i16>
  ret <8 x i16>%ex
}


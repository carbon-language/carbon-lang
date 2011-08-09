; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vcmpltps %ymm
; CHECK-NOT: vucomiss
define <8 x i32> @cmp00(<8 x float> %a, <8 x float> %b) nounwind readnone {
  %bincmp = fcmp olt <8 x float> %a, %b
  %s = sext <8 x i1> %bincmp to <8 x i32>
  ret <8 x i32> %s
}

; CHECK: vcmpltpd %ymm
; CHECK-NOT: vucomisd
define <4 x i64> @cmp01(<4 x double> %a, <4 x double> %b) nounwind readnone {
  %bincmp = fcmp olt <4 x double> %a, %b
  %s = sext <4 x i1> %bincmp to <4 x i64>
  ret <4 x i64> %s
}


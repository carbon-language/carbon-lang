; RUN: llc < %s -march=x86-64 -mattr=+avx512f | FileCheck %s

define void @f(<16 x float> %A, <16 x float> %AA, i8* %B, <8 x double> %C, <8 x double> %CC, i32 %D, <8 x i64> %E, <8 x i64> %EE) {
; CHECK: vmovntps %z
  %cast = bitcast i8* %B to <16 x float>*
  %A2 = fadd <16 x float> %A, %AA
  store <16 x float> %A2, <16 x float>* %cast, align 64, !nontemporal !0
; CHECK: vmovntdq %z
  %cast1 = bitcast i8* %B to <8 x i64>*
  %E2 = add <8 x i64> %E, %EE
  store <8 x i64> %E2, <8 x i64>* %cast1, align 64, !nontemporal !0
; CHECK: vmovntpd %z
  %cast2 = bitcast i8* %B to <8 x double>*
  %C2 = fadd <8 x double> %C, %CC
  store <8 x double> %C2, <8 x double>* %cast2, align 64, !nontemporal !0
  ret void
}

!0 = metadata !{i32 1}

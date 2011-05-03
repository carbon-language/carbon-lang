; RUN: llc < %s -march=x86 -mattr=+sse2 | FileCheck %s

define void @f(<4 x float> %A, i8* %B, <2 x double> %C, i32 %D, <2 x i64> %E) {
; CHECK: movntps
  %cast = bitcast i8* %B to <4 x float>*
  store <4 x float> %A, <4 x float>* %cast, align 16, !nontemporal !0
; CHECK: movntdq
  %cast1 = bitcast i8* %B to <2 x i64>*
  store <2 x i64> %E, <2 x i64>* %cast1, align 16, !nontemporal !0
; CHECK: movntpd
  %cast2 = bitcast i8* %B to <2 x double>*
  store <2 x double> %C, <2 x double>* %cast2, align 16, !nontemporal !0
; CHECK: movnti
  %cast3 = bitcast i8* %B to i32*
  store i32 %D, i32* %cast3, align 16, !nontemporal !0
  ret void
}

!0 = metadata !{i32 1}

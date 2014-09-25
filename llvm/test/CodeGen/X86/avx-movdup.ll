; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vmovsldup
define <8 x float> @movdupA(<8 x float> %src) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x float> %src, <8 x float> undef, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  ret <8 x float> %shuffle.i
}

; CHECK: vmovshdup
define <8 x float> @movdupB(<8 x float> %src) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x float> %src, <8 x float> undef, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  ret <8 x float> %shuffle.i
}

; CHECK: vmovsldup
define <4 x i64> @movdupC(<4 x i64> %src) nounwind uwtable readnone ssp {
entry:
  %0 = bitcast <4 x i64> %src to <8 x float>
  %shuffle.i = shufflevector <8 x float> %0, <8 x float> undef, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  %1 = bitcast <8 x float> %shuffle.i to <4 x i64>
  ret <4 x i64> %1
}

; CHECK: vmovshdup
define <4 x i64> @movdupD(<4 x i64> %src) nounwind uwtable readnone ssp {
entry:
  %0 = bitcast <4 x i64> %src to <8 x float>
  %shuffle.i = shufflevector <8 x float> %0, <8 x float> undef, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  %1 = bitcast <8 x float> %shuffle.i to <4 x i64>
  ret <4 x i64> %1
}

; Check that there is a pattern (v2f64 (X86VBroadcast f64:$src)).

; CHECK-LABEL: _vbroadcast_v128_f64
; CHECK: vmovsd LCPI{{[0-9]+}}_0(%rip), %xmm[[R0:[0-9]+]]
; CHECK: vmovddup %xmm[[R0]], %xmm{{[0-9]+}}

@E1 = external global [5 x double], align 16
@.str3 = external unnamed_addr constant [44 x i8], align 1

define void @vbroadcast_v128_f64() #0 {
entry:
  store <2 x double> <double -1.000000e+00, double -1.000000e+00>, <2 x double>* bitcast (double* getelementptr inbounds ([5 x double]* @E1, i64 0, i64 2) to <2 x double>*), align 16
  tail call void @foo1(double -1.000000e+00)
  ret void
}

declare void @foo1(double)

attributes #0 = { optsize }

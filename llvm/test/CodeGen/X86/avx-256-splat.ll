; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; FIXME: use avx versions for punpcklbw, punpckhbw and punpckhwd

; CHECK: vextractf128 $0
; CHECK-NEXT: punpcklbw
; CHECK-NEXT: punpckhbw
; CHECK-NEXT: vinsertf128 $1
; CHECK-NEXT: vpermilps $85
define <32 x i8> @funcA(<32 x i8> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> undef, <32 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  ret <32 x i8> %shuffle
}

; CHECK: vextractf128 $0
; CHECK-NEXT: punpckhwd
; CHECK-NEXT: vinsertf128 $1
; CHECK-NEXT: vpermilps $85
define <16 x i16> @funcB(<16 x i16> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> undef, <16 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  ret <16 x i16> %shuffle
}

; CHECK: vmovd
; CHECK-NEXT: movlhps
; CHECK-NEXT: vinsertf128 $1
define <4 x i64> @funcC(i64 %q) nounwind uwtable readnone ssp {
entry:
  %vecinit.i = insertelement <4 x i64> undef, i64 %q, i32 0
  %vecinit2.i = insertelement <4 x i64> %vecinit.i, i64 %q, i32 1
  %vecinit4.i = insertelement <4 x i64> %vecinit2.i, i64 %q, i32 2
  %vecinit6.i = insertelement <4 x i64> %vecinit4.i, i64 %q, i32 3
  ret <4 x i64> %vecinit6.i
}

; CHECK: vshufpd
; CHECK-NEXT: vinsertf128 $1
define <4 x double> @funcD(double %q) nounwind uwtable readnone ssp {
entry:
  %vecinit.i = insertelement <4 x double> undef, double %q, i32 0
  %vecinit2.i = insertelement <4 x double> %vecinit.i, double %q, i32 1
  %vecinit4.i = insertelement <4 x double> %vecinit2.i, double %q, i32 2
  %vecinit6.i = insertelement <4 x double> %vecinit4.i, double %q, i32 3
  ret <4 x double> %vecinit6.i
}

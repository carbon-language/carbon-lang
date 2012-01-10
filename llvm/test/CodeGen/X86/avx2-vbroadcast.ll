; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

; CHECK: vpbroadcastb (%
define <16 x i8> @BB16(i8* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i8* %ptr, align 4
  %q0 = insertelement <16 x i8> undef, i8 %q, i32 0
  %q1 = insertelement <16 x i8> %q0, i8 %q, i32 1
  %q2 = insertelement <16 x i8> %q1, i8 %q, i32 2
  %q3 = insertelement <16 x i8> %q2, i8 %q, i32 3
  %q4 = insertelement <16 x i8> %q3, i8 %q, i32 4
  %q5 = insertelement <16 x i8> %q4, i8 %q, i32 5
  %q6 = insertelement <16 x i8> %q5, i8 %q, i32 6
  %q7 = insertelement <16 x i8> %q6, i8 %q, i32 7
  %q8 = insertelement <16 x i8> %q7, i8 %q, i32 8
  %q9 = insertelement <16 x i8> %q8, i8 %q, i32 9
  %qa = insertelement <16 x i8> %q9, i8 %q, i32 10
  %qb = insertelement <16 x i8> %qa, i8 %q, i32 11
  %qc = insertelement <16 x i8> %qb, i8 %q, i32 12
  %qd = insertelement <16 x i8> %qc, i8 %q, i32 13
  %qe = insertelement <16 x i8> %qd, i8 %q, i32 14
  %qf = insertelement <16 x i8> %qe, i8 %q, i32 15
  ret <16 x i8> %qf
}
; CHECK: vpbroadcastb (%
define <32 x i8> @BB32(i8* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i8* %ptr, align 4
  %q0 = insertelement <32 x i8> undef, i8 %q, i32 0
  %q1 = insertelement <32 x i8> %q0, i8 %q, i32 1
  %q2 = insertelement <32 x i8> %q1, i8 %q, i32 2
  %q3 = insertelement <32 x i8> %q2, i8 %q, i32 3
  %q4 = insertelement <32 x i8> %q3, i8 %q, i32 4
  %q5 = insertelement <32 x i8> %q4, i8 %q, i32 5
  %q6 = insertelement <32 x i8> %q5, i8 %q, i32 6
  %q7 = insertelement <32 x i8> %q6, i8 %q, i32 7
  %q8 = insertelement <32 x i8> %q7, i8 %q, i32 8
  %q9 = insertelement <32 x i8> %q8, i8 %q, i32 9
  %qa = insertelement <32 x i8> %q9, i8 %q, i32 10
  %qb = insertelement <32 x i8> %qa, i8 %q, i32 11
  %qc = insertelement <32 x i8> %qb, i8 %q, i32 12
  %qd = insertelement <32 x i8> %qc, i8 %q, i32 13
  %qe = insertelement <32 x i8> %qd, i8 %q, i32 14
  %qf = insertelement <32 x i8> %qe, i8 %q, i32 15

  %q20 = insertelement <32 x i8> %qf, i8 %q,  i32 16
  %q21 = insertelement <32 x i8> %q20, i8 %q, i32 17
  %q22 = insertelement <32 x i8> %q21, i8 %q, i32 18
  %q23 = insertelement <32 x i8> %q22, i8 %q, i32 19
  %q24 = insertelement <32 x i8> %q23, i8 %q, i32 20
  %q25 = insertelement <32 x i8> %q24, i8 %q, i32 21
  %q26 = insertelement <32 x i8> %q25, i8 %q, i32 22
  %q27 = insertelement <32 x i8> %q26, i8 %q, i32 23
  %q28 = insertelement <32 x i8> %q27, i8 %q, i32 24
  %q29 = insertelement <32 x i8> %q28, i8 %q, i32 25
  %q2a = insertelement <32 x i8> %q29, i8 %q, i32 26
  %q2b = insertelement <32 x i8> %q2a, i8 %q, i32 27
  %q2c = insertelement <32 x i8> %q2b, i8 %q, i32 28
  %q2d = insertelement <32 x i8> %q2c, i8 %q, i32 29
  %q2e = insertelement <32 x i8> %q2d, i8 %q, i32 30
  %q2f = insertelement <32 x i8> %q2e, i8 %q, i32 31
  ret <32 x i8> %q2f
}
; CHECK: vpbroadcastw (%

define <8 x i16> @W16(i16* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i16* %ptr, align 4
  %q0 = insertelement <8 x i16> undef, i16 %q, i32 0
  %q1 = insertelement <8 x i16> %q0, i16 %q, i32 1
  %q2 = insertelement <8 x i16> %q1, i16 %q, i32 2
  %q3 = insertelement <8 x i16> %q2, i16 %q, i32 3
  %q4 = insertelement <8 x i16> %q3, i16 %q, i32 4
  %q5 = insertelement <8 x i16> %q4, i16 %q, i32 5
  %q6 = insertelement <8 x i16> %q5, i16 %q, i32 6
  %q7 = insertelement <8 x i16> %q6, i16 %q, i32 7
  ret <8 x i16> %q7
}
; CHECK: vpbroadcastw (%
define <16 x i16> @WW16(i16* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i16* %ptr, align 4
  %q0 = insertelement <16 x i16> undef, i16 %q, i32 0
  %q1 = insertelement <16 x i16> %q0, i16 %q, i32 1
  %q2 = insertelement <16 x i16> %q1, i16 %q, i32 2
  %q3 = insertelement <16 x i16> %q2, i16 %q, i32 3
  %q4 = insertelement <16 x i16> %q3, i16 %q, i32 4
  %q5 = insertelement <16 x i16> %q4, i16 %q, i32 5
  %q6 = insertelement <16 x i16> %q5, i16 %q, i32 6
  %q7 = insertelement <16 x i16> %q6, i16 %q, i32 7
  %q8 = insertelement <16 x i16> %q7, i16 %q, i32 8
  %q9 = insertelement <16 x i16> %q8, i16 %q, i32 9
  %qa = insertelement <16 x i16> %q9, i16 %q, i32 10
  %qb = insertelement <16 x i16> %qa, i16 %q, i32 11
  %qc = insertelement <16 x i16> %qb, i16 %q, i32 12
  %qd = insertelement <16 x i16> %qc, i16 %q, i32 13
  %qe = insertelement <16 x i16> %qd, i16 %q, i32 14
  %qf = insertelement <16 x i16> %qe, i16 %q, i32 15
  ret <16 x i16> %qf
}
; CHECK: vpbroadcastd (%
define <4 x i32> @D32(i32* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i32* %ptr, align 4
  %q0 = insertelement <4 x i32> undef, i32 %q, i32 0
  %q1 = insertelement <4 x i32> %q0, i32 %q, i32 1
  %q2 = insertelement <4 x i32> %q1, i32 %q, i32 2
  %q3 = insertelement <4 x i32> %q2, i32 %q, i32 3
  ret <4 x i32> %q3
}
; CHECK: vpbroadcastd (%
define <8 x i32> @DD32(i32* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i32* %ptr, align 4
  %q0 = insertelement <8 x i32> undef, i32 %q, i32 0
  %q1 = insertelement <8 x i32> %q0, i32 %q, i32 1
  %q2 = insertelement <8 x i32> %q1, i32 %q, i32 2
  %q3 = insertelement <8 x i32> %q2, i32 %q, i32 3
  %q4 = insertelement <8 x i32> %q3, i32 %q, i32 4
  %q5 = insertelement <8 x i32> %q4, i32 %q, i32 5
  %q6 = insertelement <8 x i32> %q5, i32 %q, i32 6
  %q7 = insertelement <8 x i32> %q6, i32 %q, i32 7
  ret <8 x i32> %q7
}
; CHECK: vpbroadcastq (%
define <2 x i64> @Q64(i64* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i64* %ptr, align 4
  %q0 = insertelement <2 x i64> undef, i64 %q, i32 0
  %q1 = insertelement <2 x i64> %q0, i64 %q, i32 1
  ret <2 x i64> %q1
}
; CHECK: vpbroadcastq (%
define <4 x i64> @QQ64(i64* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i64* %ptr, align 4
  %q0 = insertelement <4 x i64> undef, i64 %q, i32 0
  %q1 = insertelement <4 x i64> %q0, i64 %q, i32 1
  %q2 = insertelement <4 x i64> %q1, i64 %q, i32 2
  %q3 = insertelement <4 x i64> %q2, i64 %q, i32 3
  ret <4 x i64> %q3
}

; make sure that we still don't support broadcast double into 128-bit vector
; this used to crash
define <2 x double> @I(double* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load double* %ptr, align 4
  %vecinit.i = insertelement <2 x double> undef, double %q, i32 0
  %vecinit2.i = insertelement <2 x double> %vecinit.i, double %q, i32 1
  ret <2 x double> %vecinit2.i
}

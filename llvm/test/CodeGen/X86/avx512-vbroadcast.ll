; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl --show-mc-encoding| FileCheck %s

;CHECK-LABEL: _inreg16xi32:
;CHECK: vpbroadcastd {{.*}}, %zmm
;CHECK: ret
define   <16 x i32> @_inreg16xi32(i32 %a) {
  %b = insertelement <16 x i32> undef, i32 %a, i32 0
  %c = shufflevector <16 x i32> %b, <16 x i32> undef, <16 x i32> zeroinitializer
  ret <16 x i32> %c
}

;CHECK-LABEL: _inreg8xi64:
;CHECK: vpbroadcastq {{.*}}, %zmm
;CHECK: ret
define   <8 x i64> @_inreg8xi64(i64 %a) {
  %b = insertelement <8 x i64> undef, i64 %a, i32 0
  %c = shufflevector <8 x i64> %b, <8 x i64> undef, <8 x i32> zeroinitializer
  ret <8 x i64> %c
}

;CHECK-LABEL: _inreg16xfloat:
;CHECK: vbroadcastss {{.*}}, %zmm
;CHECK: ret
define   <16 x float> @_inreg16xfloat(float %a) {
  %b = insertelement <16 x float> undef, float %a, i32 0
  %c = shufflevector <16 x float> %b, <16 x float> undef, <16 x i32> zeroinitializer
  ret <16 x float> %c
}

;CHECK-LABEL: _inreg8xdouble:
;CHECK: vbroadcastsd {{.*}}, %zmm
;CHECK: ret
define   <8 x double> @_inreg8xdouble(double %a) {
  %b = insertelement <8 x double> undef, double %a, i32 0
  %c = shufflevector <8 x double> %b, <8 x double> undef, <8 x i32> zeroinitializer
  ret <8 x double> %c
}

;CHECK-LABEL: _xmm16xi32
;CHECK: vpbroadcastd
;CHECK: ret
define   <16 x i32> @_xmm16xi32(<16 x i32> %a) {
  %b = shufflevector <16 x i32> %a, <16 x i32> undef, <16 x i32> zeroinitializer
  ret <16 x i32> %b
}

;CHECK-LABEL: _xmm16xfloat
;CHECK: vbroadcastss {{.*}}## encoding: [0x62
;CHECK: ret
define   <16 x float> @_xmm16xfloat(<16 x float> %a) {
  %b = shufflevector <16 x float> %a, <16 x float> undef, <16 x i32> zeroinitializer
  ret <16 x float> %b
}

define <16 x i32> @test_vbroadcast() {
; CHECK-LABEL: _test_vbroadcast:
; CHECK: vpbroadcastd
entry:
  %0 = sext <16 x i1> zeroinitializer to <16 x i32>
  %1 = fcmp uno <16 x float> undef, zeroinitializer
  %2 = sext <16 x i1> %1 to <16 x i32>
  %3 = select <16 x i1> %1, <16 x i32> %0, <16 x i32> %2
  ret <16 x i32> %3
}

; We implement the set1 intrinsics with vector initializers.  Verify that the
; IR generated will produce broadcasts at the end.
define <8 x double> @test_set1_pd(double %d) #2 {
; CHECK-LABEL: _test_set1_pd:
; CHECK: vbroadcastsd
entry:
  %vecinit.i = insertelement <8 x double> undef, double %d, i32 0
  %vecinit1.i = insertelement <8 x double> %vecinit.i, double %d, i32 1
  %vecinit2.i = insertelement <8 x double> %vecinit1.i, double %d, i32 2
  %vecinit3.i = insertelement <8 x double> %vecinit2.i, double %d, i32 3
  %vecinit4.i = insertelement <8 x double> %vecinit3.i, double %d, i32 4
  %vecinit5.i = insertelement <8 x double> %vecinit4.i, double %d, i32 5
  %vecinit6.i = insertelement <8 x double> %vecinit5.i, double %d, i32 6
  %vecinit7.i = insertelement <8 x double> %vecinit6.i, double %d, i32 7
  ret <8 x double> %vecinit7.i
}

define <8 x i64> @test_set1_epi64(i64 %d) #2 {
; CHECK-LABEL: _test_set1_epi64:
; CHECK: vpbroadcastq
entry:
  %vecinit.i = insertelement <8 x i64> undef, i64 %d, i32 0
  %vecinit1.i = insertelement <8 x i64> %vecinit.i, i64 %d, i32 1
  %vecinit2.i = insertelement <8 x i64> %vecinit1.i, i64 %d, i32 2
  %vecinit3.i = insertelement <8 x i64> %vecinit2.i, i64 %d, i32 3
  %vecinit4.i = insertelement <8 x i64> %vecinit3.i, i64 %d, i32 4
  %vecinit5.i = insertelement <8 x i64> %vecinit4.i, i64 %d, i32 5
  %vecinit6.i = insertelement <8 x i64> %vecinit5.i, i64 %d, i32 6
  %vecinit7.i = insertelement <8 x i64> %vecinit6.i, i64 %d, i32 7
  ret <8 x i64> %vecinit7.i
}

define <16 x float> @test_set1_ps(float %f) #2 {
; CHECK-LABEL: _test_set1_ps:
; CHECK: vbroadcastss
entry:
  %vecinit.i = insertelement <16 x float> undef, float %f, i32 0
  %vecinit1.i = insertelement <16 x float> %vecinit.i, float %f, i32 1
  %vecinit2.i = insertelement <16 x float> %vecinit1.i, float %f, i32 2
  %vecinit3.i = insertelement <16 x float> %vecinit2.i, float %f, i32 3
  %vecinit4.i = insertelement <16 x float> %vecinit3.i, float %f, i32 4
  %vecinit5.i = insertelement <16 x float> %vecinit4.i, float %f, i32 5
  %vecinit6.i = insertelement <16 x float> %vecinit5.i, float %f, i32 6
  %vecinit7.i = insertelement <16 x float> %vecinit6.i, float %f, i32 7
  %vecinit8.i = insertelement <16 x float> %vecinit7.i, float %f, i32 8
  %vecinit9.i = insertelement <16 x float> %vecinit8.i, float %f, i32 9
  %vecinit10.i = insertelement <16 x float> %vecinit9.i, float %f, i32 10
  %vecinit11.i = insertelement <16 x float> %vecinit10.i, float %f, i32 11
  %vecinit12.i = insertelement <16 x float> %vecinit11.i, float %f, i32 12
  %vecinit13.i = insertelement <16 x float> %vecinit12.i, float %f, i32 13
  %vecinit14.i = insertelement <16 x float> %vecinit13.i, float %f, i32 14
  %vecinit15.i = insertelement <16 x float> %vecinit14.i, float %f, i32 15
  ret <16 x float> %vecinit15.i
}

define <16 x i32> @test_set1_epi32(i32 %f) #2 {
; CHECK-LABEL: _test_set1_epi32:
; CHECK: vpbroadcastd
entry:
  %vecinit.i = insertelement <16 x i32> undef, i32 %f, i32 0
  %vecinit1.i = insertelement <16 x i32> %vecinit.i, i32 %f, i32 1
  %vecinit2.i = insertelement <16 x i32> %vecinit1.i, i32 %f, i32 2
  %vecinit3.i = insertelement <16 x i32> %vecinit2.i, i32 %f, i32 3
  %vecinit4.i = insertelement <16 x i32> %vecinit3.i, i32 %f, i32 4
  %vecinit5.i = insertelement <16 x i32> %vecinit4.i, i32 %f, i32 5
  %vecinit6.i = insertelement <16 x i32> %vecinit5.i, i32 %f, i32 6
  %vecinit7.i = insertelement <16 x i32> %vecinit6.i, i32 %f, i32 7
  %vecinit8.i = insertelement <16 x i32> %vecinit7.i, i32 %f, i32 8
  %vecinit9.i = insertelement <16 x i32> %vecinit8.i, i32 %f, i32 9
  %vecinit10.i = insertelement <16 x i32> %vecinit9.i, i32 %f, i32 10
  %vecinit11.i = insertelement <16 x i32> %vecinit10.i, i32 %f, i32 11
  %vecinit12.i = insertelement <16 x i32> %vecinit11.i, i32 %f, i32 12
  %vecinit13.i = insertelement <16 x i32> %vecinit12.i, i32 %f, i32 13
  %vecinit14.i = insertelement <16 x i32> %vecinit13.i, i32 %f, i32 14
  %vecinit15.i = insertelement <16 x i32> %vecinit14.i, i32 %f, i32 15
  ret <16 x i32> %vecinit15.i
}

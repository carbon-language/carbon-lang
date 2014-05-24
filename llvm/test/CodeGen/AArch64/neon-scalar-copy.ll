; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s --check-prefix=CHECK


define float @test_dup_sv2S(<2 x float> %v) {
 ; CHECK-LABEL: test_dup_sv2S
 ; CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
 %tmp1 = extractelement <2 x float> %v, i32 1
 ret float  %tmp1
}

define float @test_dup_sv2S_0(<2 x float> %v) {
 ; CHECK-LABEL: test_dup_sv2S_0
 ; CHECK-NOT: dup {{[vsd][0-9]+}}
 ; CHECK-NOT: ins {{[vsd][0-9]+}}
 ; CHECK: ret
 %tmp1 = extractelement <2 x float> %v, i32 0
 ret float  %tmp1
}

define float @test_dup_sv4S(<4 x float> %v) {
 ; CHECK-LABEL: test_dup_sv4S
 ; CHECK-NOT: dup {{[vsd][0-9]+}}
 ; CHECK-NOT: ins {{[vsd][0-9]+}}
 ; CHECK: ret
 %tmp1 = extractelement <4 x float> %v, i32 0
 ret float  %tmp1
}

define double @test_dup_dvD(<1 x double> %v) {
 ; CHECK-LABEL: test_dup_dvD
 ; CHECK-NOT: dup {{[vsd][0-9]+}}
 ; CHECK-NOT: ins {{[vsd][0-9]+}}
 ; CHECK: ret
 %tmp1 = extractelement <1 x double> %v, i32 0
 ret double  %tmp1
}

define double @test_dup_dv2D(<2 x double> %v) {
 ; CHECK-LABEL: test_dup_dv2D
 ; CHECK: ins {{v[0-9]+}}.d[0], {{v[0-9]+}}.d[1]
 %tmp1 = extractelement <2 x double> %v, i32 1
 ret double  %tmp1
}

define double @test_dup_dv2D_0(<2 x double> %v) {
 ; CHECK-LABEL: test_dup_dv2D_0
 ; CHECK: ins {{v[0-9]+}}.d[0], {{v[0-9]+}}.d[1]
 ; CHECK: ret
 %tmp1 = extractelement <2 x double> %v, i32 1
 ret double  %tmp1
}

define <1 x i8> @test_vector_dup_bv16B(<16 x i8> %v1) {
 ; CHECK-LABEL: test_vector_dup_bv16B
 %shuffle.i = shufflevector <16 x i8> %v1, <16 x i8> undef, <1 x i32> <i32 14> 
 ret <1 x i8> %shuffle.i
}

define <1 x i8> @test_vector_dup_bv8B(<8 x i8> %v1) {
 ; CHECK-LABEL: test_vector_dup_bv8B
 %shuffle.i = shufflevector <8 x i8> %v1, <8 x i8> undef, <1 x i32> <i32 7> 
 ret <1 x i8> %shuffle.i
}

define <1 x i16> @test_vector_dup_hv8H(<8 x i16> %v1) {
 ; CHECK-LABEL: test_vector_dup_hv8H
 %shuffle.i = shufflevector <8 x i16> %v1, <8 x i16> undef, <1 x i32> <i32 7> 
 ret <1 x i16> %shuffle.i
}

define <1 x i16> @test_vector_dup_hv4H(<4 x i16> %v1) {
 ; CHECK-LABEL: test_vector_dup_hv4H
 %shuffle.i = shufflevector <4 x i16> %v1, <4 x i16> undef, <1 x i32> <i32 3> 
 ret <1 x i16> %shuffle.i
}

define <1 x i32> @test_vector_dup_sv4S(<4 x i32> %v1) {
 ; CHECK-LABEL: test_vector_dup_sv4S
 %shuffle = shufflevector <4 x i32> %v1, <4 x i32> undef, <1 x i32> <i32 3> 
 ret <1 x i32> %shuffle
}

define <1 x i32> @test_vector_dup_sv2S(<2 x i32> %v1) {
 ; CHECK-LABEL: test_vector_dup_sv2S
 %shuffle = shufflevector <2 x i32> %v1, <2 x i32> undef, <1 x i32> <i32 1> 
 ret <1 x i32> %shuffle
}

define <1 x i64> @test_vector_dup_dv2D(<2 x i64> %v1) {
 ; CHECK-LABEL: test_vector_dup_dv2D
 ; CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #8
 %shuffle.i = shufflevector <2 x i64> %v1, <2 x i64> undef, <1 x i32> <i32 1> 
 ret <1 x i64> %shuffle.i
}

define <1 x i64> @test_vector_copy_dup_dv2D(<1 x i64> %a, <2 x i64> %c) {
  ; CHECK-LABEL: test_vector_copy_dup_dv2D
  ; CHECK: {{dup|mov}} {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %vget_lane = extractelement <2 x i64> %c, i32 1
  %vset_lane = insertelement <1 x i64> undef, i64 %vget_lane, i32 0
  ret <1 x i64> %vset_lane
}


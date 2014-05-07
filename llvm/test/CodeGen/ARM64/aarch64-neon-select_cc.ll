; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-none-linux-gnu -mattr=+neon -fp-contract=fast | FileCheck %s

define <8x i8> @test_select_cc_v8i8_i8(i8 %a, i8 %b, <8x i8> %c, <8x i8> %d ) {
; CHECK-LABEL: test_select_cc_v8i8_i8:
; CHECK-DAG: fmov s[[LHS:[0-9]+]], w0
; CHECK-DAG: fmov s[[RHS:[0-9]+]], w1
; CHECK: cmeq [[MASK:v[0-9]+]].8b, v[[LHS]].8b, v[[RHS]].8b
; CHECK: dup [[DUPMASK:v[0-9]+]].8b, [[MASK]].b[0]
; CHECK: bsl [[DUPMASK]].8b, v0.8b, v1.8b
  %cmp31 = icmp eq i8 %a, %b
  %e = select i1 %cmp31, <8x i8> %c, <8x i8> %d
  ret <8x i8> %e
}

define <8x i8> @test_select_cc_v8i8_f32(float %a, float %b, <8x i8> %c, <8x i8> %d ) {
; CHECK-LABEL: test_select_cc_v8i8_f32:
; CHECK: fcmeq [[MASK:v[0-9]+]].2s, v0.2s, v1.2s
; CHECK-NEXT: dup [[DUPMASK:v[0-9]+]].2s, [[MASK]].s[0]
; CHECK-NEXT: bsl [[DUPMASK]].8b, v2.8b, v3.8b
  %cmp31 = fcmp oeq float %a, %b
  %e = select i1 %cmp31, <8x i8> %c, <8x i8> %d
  ret <8x i8> %e
}

define <8x i8> @test_select_cc_v8i8_f64(double %a, double %b, <8x i8> %c, <8x i8> %d ) {
; CHECK-LABEL: test_select_cc_v8i8_f64:
; CHECK: fcmeq d[[MASK:[0-9]+]], d0, d1
; CHECK-NEXT: bsl v[[MASK]].8b, v2.8b, v3.8b
  %cmp31 = fcmp oeq double %a, %b
  %e = select i1 %cmp31, <8x i8> %c, <8x i8> %d
  ret <8x i8> %e
}

define <16x i8> @test_select_cc_v16i8_i8(i8 %a, i8 %b, <16x i8> %c, <16x i8> %d ) {
; CHECK-LABEL: test_select_cc_v16i8_i8:
; CHECK-DAG: fmov s[[LHS:[0-9]+]], w0
; CHECK-DAG: fmov s[[RHS:[0-9]+]], w1
; CHECK: cmeq [[MASK:v[0-9]+]].16b, v[[LHS]].16b, v[[RHS]].16b
; CHECK: dup [[DUPMASK:v[0-9]+]].16b, [[MASK]].b[0]
; CHECK: bsl [[DUPMASK]].16b, v0.16b, v1.16b
  %cmp31 = icmp eq i8 %a, %b
  %e = select i1 %cmp31, <16x i8> %c, <16x i8> %d
  ret <16x i8> %e
}

define <16x i8> @test_select_cc_v16i8_f32(float %a, float %b, <16x i8> %c, <16x i8> %d ) {
; CHECK-LABEL: test_select_cc_v16i8_f32:
; CHECK: fcmeq [[MASK:v[0-9]+]].4s, v0.4s, v1.4s
; CHECK-NEXT: dup [[DUPMASK:v[0-9]+]].4s, [[MASK]].s[0]
; CHECK-NEXT: bsl [[DUPMASK]].16b, v2.16b, v3.16b
  %cmp31 = fcmp oeq float %a, %b
  %e = select i1 %cmp31, <16x i8> %c, <16x i8> %d
  ret <16x i8> %e
}

define <16x i8> @test_select_cc_v16i8_f64(double %a, double %b, <16x i8> %c, <16x i8> %d ) {
; CHECK-LABEL: test_select_cc_v16i8_f64:
; CHECK: fcmeq [[MASK:v[0-9]+]].2d, v0.2d, v1.2d
; CHECK-NEXT: dup [[DUPMASK:v[0-9]+]].2d, [[MASK]].d[0]
; CHECK-NEXT: bsl [[DUPMASK]].16b, v2.16b, v3.16b
  %cmp31 = fcmp oeq double %a, %b
  %e = select i1 %cmp31, <16x i8> %c, <16x i8> %d
  ret <16x i8> %e
}

define <4x i16> @test_select_cc_v4i16(i16 %a, i16 %b, <4x i16> %c, <4x i16> %d ) {
; CHECK-LABEL: test_select_cc_v4i16:
; CHECK-DAG: fmov s[[LHS:[0-9]+]], w0
; CHECK-DAG: fmov s[[RHS:[0-9]+]], w1
; CHECK: cmeq [[MASK:v[0-9]+]].4h, v[[LHS]].4h, v[[RHS]].4h
; CHECK: dup [[DUPMASK:v[0-9]+]].4h, [[MASK]].h[0]
; CHECK: bsl [[DUPMASK]].8b, v0.8b, v1.8b
  %cmp31 = icmp eq i16 %a, %b
  %e = select i1 %cmp31, <4x i16> %c, <4x i16> %d
  ret <4x i16> %e
}

define <8x i16> @test_select_cc_v8i16(i16 %a, i16 %b, <8x i16> %c, <8x i16> %d ) {
; CHECK-LABEL: test_select_cc_v8i16:
; CHECK-DAG: fmov s[[LHS:[0-9]+]], w0
; CHECK-DAG: fmov s[[RHS:[0-9]+]], w1
; CHECK: cmeq [[MASK:v[0-9]+]].8h, v[[LHS]].8h, v[[RHS]].8h
; CHECK: dup [[DUPMASK:v[0-9]+]].8h, [[MASK]].h[0]
; CHECK: bsl [[DUPMASK]].16b, v0.16b, v1.16b
  %cmp31 = icmp eq i16 %a, %b
  %e = select i1 %cmp31, <8x i16> %c, <8x i16> %d
  ret <8x i16> %e
}

define <2x i32> @test_select_cc_v2i32(i32 %a, i32 %b, <2x i32> %c, <2x i32> %d ) {
; CHECK-LABEL: test_select_cc_v2i32:
; CHECK-DAG: fmov s[[LHS:[0-9]+]], w0
; CHECK-DAG: fmov s[[RHS:[0-9]+]], w1
; CHECK: cmeq [[MASK:v[0-9]+]].2s, v[[LHS]].2s, v[[RHS]].2s
; CHECK: dup [[DUPMASK:v[0-9]+]].2s, [[MASK]].s[0]
; CHECK: bsl [[DUPMASK]].8b, v0.8b, v1.8b
  %cmp31 = icmp eq i32 %a, %b
  %e = select i1 %cmp31, <2x i32> %c, <2x i32> %d
  ret <2x i32> %e
}

define <4x i32> @test_select_cc_v4i32(i32 %a, i32 %b, <4x i32> %c, <4x i32> %d ) {
; CHECK-LABEL: test_select_cc_v4i32:
; CHECK-DAG: fmov s[[LHS:[0-9]+]], w0
; CHECK-DAG: fmov s[[RHS:[0-9]+]], w1
; CHECK: cmeq [[MASK:v[0-9]+]].4s, v[[LHS]].4s, v[[RHS]].4s
; CHECK: dup [[DUPMASK:v[0-9]+]].4s, [[MASK]].s[0]
; CHECK: bsl [[DUPMASK]].16b, v0.16b, v1.16b
  %cmp31 = icmp eq i32 %a, %b
  %e = select i1 %cmp31, <4x i32> %c, <4x i32> %d
  ret <4x i32> %e
}

define <1x i64> @test_select_cc_v1i64(i64 %a, i64 %b, <1x i64> %c, <1x i64> %d ) {
; CHECK-LABEL: test_select_cc_v1i64:
; CHECK-DAG: fmov d[[LHS:[0-9]+]], x0
; CHECK-DAG: fmov d[[RHS:[0-9]+]], x1
; CHECK: cmeq d[[MASK:[0-9]+]], d[[LHS]], d[[RHS]]
; CHECK: bsl v[[MASK]].8b, v0.8b, v1.8b
  %cmp31 = icmp eq i64 %a, %b
  %e = select i1 %cmp31, <1x i64> %c, <1x i64> %d
  ret <1x i64> %e
}

define <2x i64> @test_select_cc_v2i64(i64 %a, i64 %b, <2x i64> %c, <2x i64> %d ) {
; CHECK-LABEL: test_select_cc_v2i64:
; CHECK-DAG: fmov d[[LHS:[0-9]+]], x0
; CHECK-DAG: fmov d[[RHS:[0-9]+]], x1
; CHECK: cmeq [[MASK:v[0-9]+]].2d, v[[LHS]].2d, v[[RHS]].2d
; CHECK: dup [[DUPMASK:v[0-9]+]].2d, [[MASK]].d[0]
; CHECK: bsl [[DUPMASK]].16b, v0.16b, v1.16b
  %cmp31 = icmp eq i64 %a, %b
  %e = select i1 %cmp31, <2x i64> %c, <2x i64> %d
  ret <2x i64> %e
}

define <1 x float> @test_select_cc_v1f32(float %a, float %b, <1 x float> %c, <1 x float> %d ) {
; CHECK-LABEL: test_select_cc_v1f32:
; CHECK: fcmp s0, s1
; CHECK-NEXT: fcsel s0, s2, s3, eq
  %cmp31 = fcmp oeq float %a, %b
  %e = select i1 %cmp31, <1 x float> %c, <1 x float> %d
  ret <1 x float> %e
}

define <2 x float> @test_select_cc_v2f32(float %a, float %b, <2 x float> %c, <2 x float> %d ) {
; CHECK-LABEL: test_select_cc_v2f32:
; CHECK: fcmeq [[MASK:v[0-9]+]].2s, v0.2s, v1.2s
; CHECK: dup [[DUPMASK:v[0-9]+]].2s, [[MASK]].s[0]
; CHECK: bsl [[DUPMASK]].8b, v2.8b, v3.8b
  %cmp31 = fcmp oeq float %a, %b
  %e = select i1 %cmp31, <2 x float> %c, <2 x float> %d
  ret <2 x float> %e
}

define <4x float> @test_select_cc_v4f32(float %a, float %b, <4x float> %c, <4x float> %d ) {
; CHECK-LABEL: test_select_cc_v4f32:
; CHECK: fcmeq [[MASK:v[0-9]+]].4s, v0.4s, v1.4s
; CHECK: dup [[DUPMASK:v[0-9]+]].4s, [[MASK]].s[0]
; CHECK: bsl [[DUPMASK]].16b, v2.16b, v3.16b
  %cmp31 = fcmp oeq float %a, %b
  %e = select i1 %cmp31, <4x float> %c, <4x float> %d
  ret <4x float> %e
}

define <4x float> @test_select_cc_v4f32_icmp(i32 %a, i32 %b, <4x float> %c, <4x float> %d ) {
; CHECK-LABEL: test_select_cc_v4f32_icmp:
; CHECK-DAG: fmov s[[LHS:[0-9]+]], w0
; CHECK-DAG: fmov s[[RHS:[0-9]+]], w1
; CHECK: cmeq [[MASK:v[0-9]+]].4s, v[[LHS]].4s, v[[RHS]].4s
; CHECK: dup [[DUPMASK:v[0-9]+]].4s, [[MASK]].s[0]
; CHECK: bsl [[DUPMASK]].16b, v0.16b, v1.16b
  %cmp31 = icmp eq i32 %a, %b
  %e = select i1 %cmp31, <4x float> %c, <4x float> %d
  ret <4x float> %e
}

define <1 x double> @test_select_cc_v1f64(double %a, double %b, <1 x double> %c, <1 x double> %d ) {
; CHECK-LABEL: test_select_cc_v1f64:
; CHECK: fcmeq d[[MASK:[0-9]+]], d0, d1
; CHECK: bsl v[[MASK]].8b, v2.8b, v3.8b
  %cmp31 = fcmp oeq double %a, %b
  %e = select i1 %cmp31, <1 x double> %c, <1 x double> %d
  ret <1 x double> %e
}

define <1 x double> @test_select_cc_v1f64_icmp(i64 %a, i64 %b, <1 x double> %c, <1 x double> %d ) {
; CHECK-LABEL: test_select_cc_v1f64_icmp:
; CHECK-DAG: fmov [[LHS:d[0-9]+]], x0
; CHECK-DAG: fmov [[RHS:d[0-9]+]], x1
; CHECK: cmeq d[[MASK:[0-9]+]], [[LHS]], [[RHS]]
; CHECK: bsl v[[MASK]].8b, v0.8b, v1.8b
  %cmp31 = icmp eq i64 %a, %b
  %e = select i1 %cmp31, <1 x double> %c, <1 x double> %d
  ret <1 x double> %e
}

define <2 x double> @test_select_cc_v2f64(double %a, double %b, <2 x double> %c, <2 x double> %d ) {
; CHECK-LABEL: test_select_cc_v2f64:
; CHECK: fcmeq [[MASK:v[0-9]+]].2d, v0.2d, v1.2d
; CHECK: dup [[DUPMASK:v[0-9]+]].2d, [[MASK]].d[0]
; CHECK: bsl [[DUPMASK]].16b, v2.16b, v3.16b
  %cmp31 = fcmp oeq double %a, %b
  %e = select i1 %cmp31, <2 x double> %c, <2 x double> %d
  ret <2 x double> %e
}

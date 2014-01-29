; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon -fp-contract=fast | FileCheck %s

define <8x i8> @test_select_cc_v8i8_i8(i8 %a, i8 %b, <8x i8> %c, <8x i8> %d ) {
; CHECK-LABEL: test_select_cc_v8i8_i8:
; CHECK: and	w0, w0, #0xff
; CHECK-NEXT: cmp	w0, w1, uxtb
; CHECK-NEXT: csinv	w0, wzr, wzr, ne
; CHECK-NEXT: dup	v{{[0-9]+}}.8b, w0
; CHECK-NEXT:	bsl	v{{[0-9]+}}.8b, v0.8b, v1.8b
  %cmp31 = icmp eq i8 %a, %b
  %e = select i1 %cmp31, <8x i8> %c, <8x i8> %d
  ret <8x i8> %e
}

define <8x i8> @test_select_cc_v8i8_f32(float %a, float %b, <8x i8> %c, <8x i8> %d ) {
; CHECK-LABEL: test_select_cc_v8i8_f32:
; CHECK: fcmeq	v{{[0-9]+}}.4s, v0.4s, v1.4s
; CHECK-NEXT: dup	v{{[0-9]+}}.2s, v{{[0-9]+}}.s[0]
; CHECK-NEXT:	bsl	v{{[0-9]+}}.8b, v2.8b, v3.8b
  %cmp31 = fcmp oeq float %a, %b
  %e = select i1 %cmp31, <8x i8> %c, <8x i8> %d
  ret <8x i8> %e
}

define <8x i8> @test_select_cc_v8i8_f64(double %a, double %b, <8x i8> %c, <8x i8> %d ) {
; CHECK-LABEL: test_select_cc_v8i8_f64:
; CHECK: fcmeq	v{{[0-9]+}}.2d, v0.2d, v1.2d
; CHECK-NEXT:	bsl	v{{[0-9]+}}.8b, v2.8b, v3.8b
  %cmp31 = fcmp oeq double %a, %b
  %e = select i1 %cmp31, <8x i8> %c, <8x i8> %d
  ret <8x i8> %e
}

define <16x i8> @test_select_cc_v16i8_i8(i8 %a, i8 %b, <16x i8> %c, <16x i8> %d ) {
; CHECK-LABEL: test_select_cc_v16i8_i8:
; CHECK: and	w0, w0, #0xff
; CHECK-NEXT: cmp	w0, w1, uxtb
; CHECK-NEXT: csinv	w0, wzr, wzr, ne
; CHECK-NEXT: dup	v{{[0-9]+}}.16b, w0
; CHECK-NEXT:	bsl	v{{[0-9]+}}.16b, v0.16b, v1.16b
  %cmp31 = icmp eq i8 %a, %b
  %e = select i1 %cmp31, <16x i8> %c, <16x i8> %d
  ret <16x i8> %e
}

define <16x i8> @test_select_cc_v16i8_f32(float %a, float %b, <16x i8> %c, <16x i8> %d ) {
; CHECK-LABEL: test_select_cc_v16i8_f32:
; CHECK: fcmeq	v{{[0-9]+}}.4s, v0.4s, v1.4s
; CHECK-NEXT: dup	v{{[0-9]+}}.4s, v{{[0-9]+}}.s[0]
; CHECK-NEXT:	bsl	v{{[0-9]+}}.16b, v2.16b, v3.16b
  %cmp31 = fcmp oeq float %a, %b
  %e = select i1 %cmp31, <16x i8> %c, <16x i8> %d
  ret <16x i8> %e
}

define <16x i8> @test_select_cc_v16i8_f64(double %a, double %b, <16x i8> %c, <16x i8> %d ) {
; CHECK-LABEL: test_select_cc_v16i8_f64:
; CHECK: fcmeq	v{{[0-9]+}}.2d, v0.2d, v1.2d
; CHECK-NEXT: dup	v{{[0-9]+}}.2d, v{{[0-9]+}}.d[0]
; CHECK-NEXT:	bsl	v{{[0-9]+}}.16b, v2.16b, v3.16b
  %cmp31 = fcmp oeq double %a, %b
  %e = select i1 %cmp31, <16x i8> %c, <16x i8> %d
  ret <16x i8> %e
}

define <4x i16> @test_select_cc_v4i16(i16 %a, i16 %b, <4x i16> %c, <4x i16> %d ) {
; CHECK-LABEL: test_select_cc_v4i16:
; CHECK: and	w0, w0, #0xffff
; CHECK-NEXT: cmp	w0, w1, uxth
; CHECK-NEXT: csinv	w0, wzr, wzr, ne
; CHECK-NEXT: dup	v{{[0-9]+}}.4h, w0
; CHECK-NEXT:	bsl	v{{[0-9]+}}.8b, v0.8b, v1.8b
  %cmp31 = icmp eq i16 %a, %b
  %e = select i1 %cmp31, <4x i16> %c, <4x i16> %d
  ret <4x i16> %e
}

define <8x i16> @test_select_cc_v8i16(i16 %a, i16 %b, <8x i16> %c, <8x i16> %d ) {
; CHECK-LABEL: test_select_cc_v8i16:
; CHECK: and	w0, w0, #0xffff
; CHECK-NEXT: cmp	w0, w1, uxth
; CHECK-NEXT: csinv	w0, wzr, wzr, ne
; CHECK-NEXT: dup	v{{[0-9]+}}.8h, w0
; CHECK-NEXT:	bsl	v{{[0-9]+}}.16b, v0.16b, v1.16b
  %cmp31 = icmp eq i16 %a, %b
  %e = select i1 %cmp31, <8x i16> %c, <8x i16> %d
  ret <8x i16> %e
}

define <2x i32> @test_select_cc_v2i32(i32 %a, i32 %b, <2x i32> %c, <2x i32> %d ) {
; CHECK-LABEL: test_select_cc_v2i32:
; CHECK: cmp	w0, w1, uxtw
; CHECK-NEXT: csinv	w0, wzr, wzr, ne
; CHECK-NEXT: dup	v{{[0-9]+}}.2s, w0
; CHECK-NEXT:	bsl	v{{[0-9]+}}.8b, v0.8b, v1.8b
  %cmp31 = icmp eq i32 %a, %b
  %e = select i1 %cmp31, <2x i32> %c, <2x i32> %d
  ret <2x i32> %e
}

define <4x i32> @test_select_cc_v4i32(i32 %a, i32 %b, <4x i32> %c, <4x i32> %d ) {
; CHECK-LABEL: test_select_cc_v4i32:
; CHECK: cmp	w0, w1, uxtw
; CHECK-NEXT: csinv	w0, wzr, wzr, ne
; CHECK-NEXT: dup	v{{[0-9]+}}.4s, w0
; CHECK-NEXT:	bsl	v{{[0-9]+}}.16b, v0.16b, v1.16b
  %cmp31 = icmp eq i32 %a, %b
  %e = select i1 %cmp31, <4x i32> %c, <4x i32> %d
  ret <4x i32> %e
}

define <1x i64> @test_select_cc_v1i64(i64 %a, i64 %b, <1x i64> %c, <1x i64> %d ) {
; CHECK-LABEL: test_select_cc_v1i64:
; CHECK: cmp	x0, x1
; CHECK-NEXT: csinv	x0, xzr, xzr, ne
; CHECK-NEXT: fmov	d{{[0-9]+}}, x0
; CHECK-NEXT:	bsl	v{{[0-9]+}}.8b, v0.8b, v1.8b
  %cmp31 = icmp eq i64 %a, %b
  %e = select i1 %cmp31, <1x i64> %c, <1x i64> %d
  ret <1x i64> %e
}

define <2x i64> @test_select_cc_v2i64(i64 %a, i64 %b, <2x i64> %c, <2x i64> %d ) {
; CHECK-LABEL: test_select_cc_v2i64:
; CHECK: cmp	x0, x1
; CHECK-NEXT: csinv	x0, xzr, xzr, ne
; CHECK-NEXT: dup	v{{[0-9]+}}.2d, x0
; CHECK-NEXT:	bsl	v{{[0-9]+}}.16b, v0.16b, v1.16b
  %cmp31 = icmp eq i64 %a, %b
  %e = select i1 %cmp31, <2x i64> %c, <2x i64> %d
  ret <2x i64> %e
}

define <1 x float> @test_select_cc_v1f32(float %a, float %b, <1 x float> %c, <1 x float> %d ) {
; CHECK-LABEL: test_select_cc_v1f32:
; CHECK: fcmp	s0, s1
; CHECK-NEXT: fcsel	s0, s2, s3, eq
  %cmp31 = fcmp oeq float %a, %b
  %e = select i1 %cmp31, <1 x float> %c, <1 x float> %d
  ret <1 x float> %e
}
  
define <2 x float> @test_select_cc_v2f32(float %a, float %b, <2 x float> %c, <2 x float> %d ) {
; CHECK-LABEL: test_select_cc_v2f32:
; CHECK: fcmeq	v{{[0-9]+}}.4s, v0.4s, v1.4s
; CHECK-NEXT: dup	v{{[0-9]+}}.2s, v{{[0-9]+}}.s[0]
; CHECK-NEXT:	bsl	v{{[0-9]+}}.8b, v2.8b, v3.8b
  %cmp31 = fcmp oeq float %a, %b
  %e = select i1 %cmp31, <2 x float> %c, <2 x float> %d
  ret <2 x float> %e
}

define <4x float> @test_select_cc_v4f32(float %a, float %b, <4x float> %c, <4x float> %d ) {
; CHECK-LABEL: test_select_cc_v4f32:
; CHECK: fcmeq	v{{[0-9]+}}.4s, v0.4s, v1.4s
; CHECK-NEXT: dup	v{{[0-9]+}}.4s, v{{[0-9]+}}.s[0]
; CHECK-NEXT:	bsl	v{{[0-9]+}}.16b, v2.16b, v3.16b
  %cmp31 = fcmp oeq float %a, %b
  %e = select i1 %cmp31, <4x float> %c, <4x float> %d
  ret <4x float> %e
}

define <1 x double> @test_select_cc_v1f64(double %a, double %b, <1 x double> %c, <1 x double> %d ) {
; CHECK-LABEL: test_select_cc_v1f64:
; CHECK: fcmeq	v{{[0-9]+}}.2d, v0.2d, v1.2d
; CHECK-NEXT:	bsl	v{{[0-9]+}}.8b, v2.8b, v3.8b
  %cmp31 = fcmp oeq double %a, %b
  %e = select i1 %cmp31, <1 x double> %c, <1 x double> %d
  ret <1 x double> %e
}

define <2 x double> @test_select_cc_v2f64(double %a, double %b, <2 x double> %c, <2 x double> %d ) {
; CHECK-LABEL: test_select_cc_v2f64:
; CHECK: fcmeq	v{{[0-9]+}}.2d, v0.2d, v1.2d
; CHECK-NEXT: dup	v{{[0-9]+}}.2d, v{{[0-9]+}}.d[0]
; CHECK-NEXT:	bsl	v{{[0-9]+}}.16b, v2.16b, v3.16b
  %cmp31 = fcmp oeq double %a, %b
  %e = select i1 %cmp31, <2 x double> %c, <2 x double> %d
  ret <2 x double> %e
}

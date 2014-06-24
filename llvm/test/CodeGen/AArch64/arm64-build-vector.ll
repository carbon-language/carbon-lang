; RUN: llc < %s -march=arm64 -aarch64-neon-syntax=apple | FileCheck %s

; Check that building up a vector w/ only one non-zero lane initializes
; intelligently.
define void @one_lane(i32* nocapture %out_int, i32 %skip0) nounwind {
; CHECK-LABEL: one_lane:
; CHECK: dup.16b v[[REG:[0-9]+]], wzr
; CHECK-NEXT: ins.b v[[REG]][0], w1
; v and q are aliases, and str is preferred against st.16b when possible
; rdar://11246289
; CHECK: str q[[REG]], [x0]
; CHECK: ret
  %conv = trunc i32 %skip0 to i8
  %vset_lane = insertelement <16 x i8> <i8 undef, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, i8 %conv, i32 0
  %tmp = bitcast i32* %out_int to <4 x i32>*
  %tmp1 = bitcast <16 x i8> %vset_lane to <4 x i32>
  store <4 x i32> %tmp1, <4 x i32>* %tmp, align 16
  ret void
}

; Check that building a vector from floats doesn't insert an unnecessary
; copy for lane zero.
define <4 x float>  @foo(float %a, float %b, float %c, float %d) nounwind {
; CHECK-LABEL: foo:
; CHECK-NOT: ins.s v0[0], v0[0]
; CHECK: ins.s v0[1], v1[0]
; CHECK: ins.s v0[2], v2[0]
; CHECK: ins.s v0[3], v3[0]
; CHECK: ret
  %1 = insertelement <4 x float> undef, float %a, i32 0
  %2 = insertelement <4 x float> %1, float %b, i32 1
  %3 = insertelement <4 x float> %2, float %c, i32 2
  %4 = insertelement <4 x float> %3, float %d, i32 3
  ret <4 x float> %4
}

define <8 x i16> @build_all_zero(<8 x i16> %a) #1 {
; CHECK-LABEL: build_all_zero:
; CHECK: movn	w[[GREG:[0-9]+]], #0x517f
; CHECK-NEXT:	fmov	s[[FREG:[0-9]+]], w[[GREG]]
; CHECK-NEXT:	mul.8h	v0, v0, v[[FREG]]
  %b = add <8 x i16> %a, <i16 -32768, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef>
  %c = mul <8 x i16> %b, <i16 -20864, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef>
  ret <8 x i16> %c
  }

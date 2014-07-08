; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 | FileCheck %s

; Test that we correctly fold a shuffle that performs a swizzle of another
; shuffle node according to the rule
;  shuffle (shuffle (x, undef, M0), undef, M1) -> shuffle(x, undef, M2)
;
; We only do this if the resulting mask is legal to avoid introducing an
; illegal shuffle that is expanded into a sub-optimal sequence of instructions
; during lowering stage.


define <4 x i32> @swizzle_1(<4 x i32> %v) {
  %1 = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 0, i32 1>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 0, i32 1>
  ret <4 x i32> %2
}
; CHECK-LABEL: swizzle_1
; Mask: [1,0,3,2]
; CHECK: pshufd $-79
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @swizzle_2(<4 x i32> %v) {
  %1 = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 3, i32 1, i32 0, i32 2>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 3, i32 1, i32 0, i32 2>
  ret <4 x i32> %2
}
; CHECK-LABEL: swizzle_2
; Mask: [2,1,3,0]
; CHECK: pshufd $54
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @swizzle_3(<4 x i32> %v) {
  %1 = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 1, i32 0>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 1, i32 0>
  ret <4 x i32> %2
}
; CHECK-LABEL: swizzle_3
; Mask: [1,0,3,2]
; CHECK: pshufd $-79
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @swizzle_4(<4 x i32> %v) {
  %1 = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 2, i32 1, i32 3, i32 0>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 2, i32 1, i32 3, i32 0>
  ret <4 x i32> %2
}
; CHECK-LABEL: swizzle_4
; Mask: [3,1,0,2]
; CHECK: pshufd $-121
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @swizzle_5(<4 x i32> %v) {
  %1 = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  ret <4 x i32> %2
}
; CHECK-LABEL: swizzle_5
; Mask: [2,3,0,1]
; CHECK: pshufd $78
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @swizzle_6(<4 x i32> %v) {
  %1 = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 1, i32 2, i32 0, i32 3>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 1, i32 2, i32 0, i32 3>
  ret <4 x i32> %2
}
; CHECK-LABEL: swizzle_6
; Mask: [2,0,1,3]
; CHECK: pshufd $-46
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @swizzle_7(<4 x i32> %v) {
  %1 = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 0, i32 3, i32 1, i32 2>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 0, i32 3, i32 1, i32 2>
  ret <4 x i32> %2
}
; CHECK-LABEL: swizzle_7
; Mask: [0,2,3,1]
; CHECK: pshufd $120
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @swizzle_8(<4 x i32> %v) {
  %1 = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 3, i32 0, i32 2, i32 1>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 3, i32 0, i32 2, i32 1>
  ret <4 x i32> %2
}
; CHECK-LABEL: swizzle_8
; Mask: [1,3,2,0]
; CHECK: pshufd $45
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @swizzle_9(<4 x i32> %v) {
  %1 = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 3, i32 0, i32 1, i32 2>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 3, i32 0, i32 1, i32 2>
  ret <4 x i32> %2
}
; CHECK-LABEL: swizzle_9
; Mask: [2,3,0,1]
; CHECK: pshufd $78
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @swizzle_10(<4 x i32> %v) {
  %1 = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 2, i32 0, i32 1, i32 3>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 2, i32 0, i32 1, i32 3>
  ret <4 x i32> %2
}
; CHECK-LABEL: swizzle_10
; Mask: [1,2,0,3]
; CHECK: pshufd $-55
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @swizzle_11(<4 x i32> %v) {
  %1 = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 2, i32 0, i32 3, i32 1>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 2, i32 0, i32 3, i32 1>
  ret <4 x i32> %2
}
; CHECK-LABEL: swizzle_11
; Mask: [3,2,1,0]
; CHECK: pshufd $27
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @swizzle_12(<4 x i32> %v) {
  %1 = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 0, i32 2, i32 3, i32 1>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 0, i32 2, i32 3, i32 1>
  ret <4 x i32> %2
}
; CHECK-LABEL: swizzle_12
; Mask: [0,3,1,2]
; CHECK: pshufd $-100
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @swizzle_13(<4 x i32> %v) {
  %1 = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 1, i32 3, i32 0, i32 2>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 1, i32 3, i32 0, i32 2>
  ret <4 x i32> %2
}
; CHECK-LABEL: swizzle_13
; Mask: [3,2,1,0]
; CHECK: pshufd $27
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @swizzle_14(<4 x i32> %v) {
  %1 = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 1, i32 3, i32 2, i32 0>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 1, i32 3, i32 2, i32 0>
  ret <4 x i32> %2
}
; CHECK-LABEL: swizzle_14
; Mask: [3,0,2,1]
; CHECK: pshufd $99
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x float> @swizzle_15(<4 x float> %v) {
  %1 = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 0, i32 1>
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 0, i32 1>
  ret <4 x float> %2
}
; CHECK-LABEL: swizzle_15
; Mask: [1,0,3,2]
; CHECK: pshufd $-79
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x float> @swizzle_16(<4 x float> %v) {
  %1 = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 3, i32 1, i32 0, i32 2>
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 3, i32 1, i32 0, i32 2>
  ret <4 x float> %2
}
; CHECK-LABEL: swizzle_16
; Mask: [2,1,3,0]
; CHECK: pshufd $54
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x float> @swizzle_17(<4 x float> %v) {
  %1 = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 2, i32 3, i32 1, i32 0>
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 2, i32 3, i32 1, i32 0>
  ret <4 x float> %2
}
; CHECK-LABEL: swizzle_17
; Mask: [1,0,3,2]
; CHECK: pshufd $-79
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x float> @swizzle_18(<4 x float> %v) {
  %1 = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 2, i32 1, i32 3, i32 0>
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 2, i32 1, i32 3, i32 0>
  ret <4 x float> %2
}
; CHECK-LABEL: swizzle_18
; Mask: [3,1,0,2]
; CHECK: pshufd $-121
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x float> @swizzle_19(<4 x float> %v) {
  %1 = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  ret <4 x float> %2
}
; CHECK-LABEL: swizzle_19
; Mask: [2,3,0,1]
; CHECK: pshufd $78
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x float> @swizzle_20(<4 x float> %v) {
  %1 = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 1, i32 2, i32 0, i32 3>
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 1, i32 2, i32 0, i32 3>
  ret <4 x float> %2
}
; CHECK-LABEL: swizzle_20
; Mask: [2,0,1,3]
; CHECK: pshufd $-46
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x float> @swizzle_21(<4 x float> %v) {
  %1 = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 0, i32 3, i32 1, i32 2>
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 0, i32 3, i32 1, i32 2>
  ret <4 x float> %2
}
; CHECK-LABEL: swizzle_21
; Mask: [0,2,3,1]
; CHECK: pshufd $120
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x float> @swizzle_22(<4 x float> %v) {
  %1 = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 3, i32 0, i32 2, i32 1>
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 3, i32 0, i32 2, i32 1>
  ret <4 x float> %2
}
; CHECK-LABEL: swizzle_22
; Mask: [1,3,2,0]
; CHECK: pshufd $45
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x float> @swizzle_23(<4 x float> %v) {
  %1 = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 3, i32 0, i32 1, i32 2>
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 3, i32 0, i32 1, i32 2>
  ret <4 x float> %2
}
; CHECK-LABEL: swizzle_23
; Mask: [2,3,0,1]
; CHECK: pshufd $78
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x float> @swizzle_24(<4 x float> %v) {
  %1 = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 2, i32 0, i32 1, i32 3>
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 2, i32 0, i32 1, i32 3>
  ret <4 x float> %2
}
; CHECK-LABEL: swizzle_24
; Mask: [1,2,0,3]
; CHECK: pshufd $-55
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x float> @swizzle_25(<4 x float> %v) {
  %1 = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 2, i32 0, i32 3, i32 1>
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 2, i32 0, i32 3, i32 1>
  ret <4 x float> %2
}
; CHECK-LABEL: swizzle_25
; Mask: [3,2,1,0]
; CHECK: pshufd $27
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x float> @swizzle_26(<4 x float> %v) {
  %1 = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 0, i32 2, i32 3, i32 1>
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 0, i32 2, i32 3, i32 1>
  ret <4 x float> %2
}
; CHECK-LABEL: swizzle_26
; Mask: [0,3,1,2]
; CHECK: pshufd $-100
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x float> @swizzle_27(<4 x float> %v) {
  %1 = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 1, i32 3, i32 0, i32 2>
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 1, i32 3, i32 0, i32 2>
  ret <4 x float> %2
}
; CHECK-LABEL: swizzle_27
; Mask: [3,2,1,0]
; CHECK: pshufd $27
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x float> @swizzle_28(<4 x float> %v) {
  %1 = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 1, i32 3, i32 2, i32 0>
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 1, i32 3, i32 2, i32 0>
  ret <4 x float> %2
}
; CHECK-LABEL: swizzle_28
; Mask: [3,0,2,1]
; CHECK: pshufd $99
; CHECK-NOT: pshufd
; CHECK-NEXT: ret


define <4 x float> @swizzle_29(<4 x float> %v) {
  %1 = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> <i32 3, i32 1, i32 2, i32 0>
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 1, i32 0, i32 2, i32 3>
  ret <4 x float> %2
}
; CHECK-LABEL: swizzle_29
; Mask: [1,3,2,0]
; CHECK: pshufd $45
; CHECK-NOT: pshufd
; CHECK-NEXT: ret

; Make sure that we combine the shuffles from each function below into a single
; legal shuffle (either pshuflw or pshufb depending on the masks).

define <8 x i16> @swizzle_30(<8 x i16> %v) {
  %1 = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 3, i32 1, i32 2, i32 0, i32 7, i32 5, i32 6, i32 4>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 1, i32 0, i32 2, i32 3, i32 7, i32 5, i32 6, i32 4>
  ret <8 x i16> %2
}
; CHECK-LABEL: swizzle_30
; Mask: [1,3,2,0,5,7,6,4]
; CHECK: pshuflw $45
; CHECK-NOT: pshufb
; CHECK-NEXT: ret


define <8 x i16> @swizzle_31(<8 x i16> %v) {
  %1 = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 3, i32 0, i32 2, i32 1, i32 7, i32 5, i32 6, i32 4>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 3, i32 0, i32 2, i32 1, i32 7, i32 5, i32 6, i32 4>
  ret <8 x i16> %2
}
; CHECK-LABEL: swizzle_31
; Mask: [1,3,2,0,4,5,6,7]
; CHECK: pshuflw $45
; CHECK-NOT: pshufb
; CHECK: ret


define <8 x i16> @swizzle_32(<8 x i16> %v) {
  %1 = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 1, i32 2, i32 3, i32 0, i32 7, i32 5, i32 6, i32 4>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 1, i32 2, i32 3, i32 0, i32 7, i32 5, i32 6, i32 4>
  ret <8 x i16> %2
}
; CHECK-LABEL: swizzle_32
; Mask: [2,3,0,1,4,5,6,7] --> equivalent to pshufd mask [1,0,2,3]
; CHECK: pshufd $-31
; CHECK-NOT: pshufb
; CHECK: ret

define <8 x i16> @swizzle_33(<8 x i16> %v) {
  %1 = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 4, i32 6, i32 5, i32 7, i32 2, i32 3, i32 1, i32 0>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 4, i32 6, i32 5, i32 7, i32 2, i32 3, i32 1, i32 0>
  ret <8 x i16> %2
}
; CHECK-LABEL: swizzle_33
; CHECK: pshufb
; CHECK-NOT: pshufb
; CHECK-NOT: shufpd
; CHECK: ret


define <8 x i16> @swizzle_34(<8 x i16> %v) {
  %1 = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 4, i32 7, i32 6, i32 5, i32 1, i32 2, i32 0, i32 3>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 4, i32 7, i32 6, i32 5, i32 1, i32 2, i32 0, i32 3>
  ret <8 x i16> %2
}
; CHECK-LABEL: swizzle_34
; CHECK: pshufb
; CHECK-NOT: pshufb
; CHECK-NOT: shufpd
; CHECK: ret


define <8 x i16> @swizzle_35(<8 x i16> %v) {
  %1 = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 7, i32 4, i32 6, i32 5, i32 1, i32 3, i32 0, i32 2>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 7, i32 4, i32 6, i32 5, i32 1, i32 3, i32 0, i32 2>
  ret <8 x i16> %2
}
; CHECK-LABEL: swizzle_35
; CHECK: pshufb
; CHECK-NOT: pshufb
; CHECK: ret


define <8 x i16> @swizzle_36(<8 x i16> %v) {
  %1 = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 4, i32 6, i32 7, i32 5, i32 0, i32 1, i32 3, i32 2>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 4, i32 6, i32 7, i32 5, i32 0, i32 1, i32 3, i32 2>
  ret <8 x i16> %2
}
; CHECK-LABEL: swizzle_36
; CHECK: pshufb
; CHECK-NOT: pshufb
; CHECK-NOT: shufpd
; CHECK: ret


define <8 x i16> @swizzle_37(<8 x i16> %v) {
  %1 = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 7, i32 5, i32 6, i32 4>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 7, i32 4, i32 6, i32 5>
  ret <8 x i16> %2
}
; CHECK-LABEL: swizzle_37
; Mask: [0,1,2,3,4,7,6,5]
; CHECK: pshufhw $108
; CHECK-NOT: pshufb
; CHECK: ret


define <8 x i16> @swizzle_38(<8 x i16> %v) {
  %1 = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 5, i32 6, i32 4, i32 7, i32 0, i32 2, i32 1, i32 3>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 5, i32 6, i32 4, i32 7, i32 0, i32 2, i32 1, i32 3>
  ret <8 x i16> %2
}
; CHECK-LABEL: swizzle_38
; CHECK: pshufb
; CHECK-NOT: pshufb
; CHECK-NOT: shufpd
; CHECK: ret


define <8 x i16> @swizzle_39(<8 x i16> %v) {
  %1 = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 5, i32 4, i32 6, i32 7, i32 3, i32 2, i32 1, i32 0>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 5, i32 4, i32 6, i32 7, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i16> %2
}
; CHECK-LABEL: swizzle_39
; CHECK: pshufb
; CHECK-NOT: pshufb
; CHECK-NOT: shufpd
; CHECK: ret


define <8 x i16> @swizzle_40(<8 x i16> %v) {
  %1 = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 6, i32 4, i32 7, i32 5, i32 1, i32 0, i32 3, i32 2>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 6, i32 4, i32 7, i32 5, i32 1, i32 0, i32 3, i32 2>
  ret <8 x i16> %2
}
; CHECK-LABEL: swizzle_40
; CHECK: pshufb
; CHECK-NOT: pshufb
; CHECK-NOT: shufpd
; CHECK: ret


define <8 x i16> @swizzle_41(<8 x i16> %v) {
  %1 = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 6, i32 7, i32 5, i32 4, i32 0, i32 1, i32 3, i32 2>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 6, i32 7, i32 5, i32 4, i32 0, i32 1, i32 3, i32 2>
  ret <8 x i16> %2
}
; CHECK-LABEL: swizzle_41
; CHECK: pshufb
; CHECK-NOT: pshufb
; CHECK-NOT: shufpd
; CHECK: ret


define <8 x i16> @swizzle_42(<8 x i16> %v) {
  %1 = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 0, i32 1, i32 3, i32 2, i32 7, i32 6, i32 4, i32 5>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 0, i32 1, i32 3, i32 2, i32 7, i32 6, i32 4, i32 5>
  ret <8 x i16> %2
}
; CHECK-LABEL: swizzle_42
; Mask: [0,1,2,3,5,4,7,6]
; CHECK: pshufhw $-79
; CHECK-NOT: pshufb
; CHECK: ret



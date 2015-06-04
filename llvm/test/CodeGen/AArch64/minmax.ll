; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s

; CHECK-LABEL: t1
; CHECK: smax
define <4 x i32> @t1(<4 x i32> %a, <4 x i32> %b) {
  %t1 = icmp sgt <4 x i32> %a, %b
  %t2 = select <4 x i1> %t1, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %t2
}

; CHECK-LABEL: t2
; CHECK: smin
define <4 x i32> @t2(<4 x i32> %a, <4 x i32> %b) {
  %t1 = icmp slt <4 x i32> %a, %b
  %t2 = select <4 x i1> %t1, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %t2
}

; CHECK-LABEL: t3
; CHECK: umax
define <4 x i32> @t3(<4 x i32> %a, <4 x i32> %b) {
  %t1 = icmp ugt <4 x i32> %a, %b
  %t2 = select <4 x i1> %t1, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %t2
}

; CHECK-LABEL: t4
; CHECK: umin
define <8 x i8> @t4(<8 x i8> %a, <8 x i8> %b) {
  %t1 = icmp ult <8 x i8> %a, %b
  %t2 = select <8 x i1> %t1, <8 x i8> %a, <8 x i8> %b
  ret <8 x i8> %t2
}

; CHECK-LABEL: t5
; CHECK: smin
define <4 x i16> @t5(<4 x i16> %a, <4 x i16> %b) {
  %t1 = icmp sgt <4 x i16> %b, %a
  %t2 = select <4 x i1> %t1, <4 x i16> %a, <4 x i16> %b
  ret <4 x i16> %t2
}

; CHECK-LABEL: t6
; CHECK: smax
define <2 x i32> @t6(<2 x i32> %a, <2 x i32> %b) {
  %t1 = icmp slt <2 x i32> %b, %a
  %t2 = select <2 x i1> %t1, <2 x i32> %a, <2 x i32> %b
  ret <2 x i32> %t2
}

; CHECK-LABEL: t7
; CHECK: umin
define <16 x i8> @t7(<16 x i8> %a, <16 x i8> %b) {
  %t1 = icmp ugt <16 x i8> %b, %a
  %t2 = select <16 x i1> %t1, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %t2
}

; CHECK-LABEL: t8
; CHECK: umax
define <8 x i16> @t8(<8 x i16> %a, <8 x i16> %b) {
  %t1 = icmp ult <8 x i16> %b, %a
  %t2 = select <8 x i1> %t1, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %t2
}

; CHECK-LABEL: t9
; CHECK: umin
; CHECK: smax
define <4 x i32> @t9(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %t1 = icmp ugt <4 x i32> %b, %a
  %t2 = select <4 x i1> %t1, <4 x i32> %a, <4 x i32> %b
  %t3 = icmp sge <4 x i32> %t2, %c
  %t4 = select <4 x i1> %t3, <4 x i32> %t2, <4 x i32> %c
  ret <4 x i32> %t4
}

; CHECK-LABEL: t10
; CHECK: smax
; CHECK: smax
define <8 x i32> @t10(<8 x i32> %a, <8 x i32> %b) {
  %t1 = icmp sgt <8 x i32> %a, %b
  %t2 = select <8 x i1> %t1, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %t2
}

; CHECK-LABEL: t11
; CHECK: smin
; CHECK: smin
; CHECK: smin
; CHECK: smin
define <16 x i32> @t11(<16 x i32> %a, <16 x i32> %b) {
  %t1 = icmp sle <16 x i32> %a, %b
  %t2 = select <16 x i1> %t1, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %t2
}

; CHECK-LABEL: t12
; CHECK-NOT: umin
; The icmp is used by two instructions, so don't produce a umin node.
define <16 x i8> @t12(<16 x i8> %a, <16 x i8> %b) {
  %t1 = icmp ugt <16 x i8> %b, %a
  %t2 = select <16 x i1> %t1, <16 x i8> %a, <16 x i8> %b
  %t3 = zext <16 x i1> %t1 to <16 x i8>
  %t4 = add <16 x i8> %t3, %t2
  ret <16 x i8> %t4
}

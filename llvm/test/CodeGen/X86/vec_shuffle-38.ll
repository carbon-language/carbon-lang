; RUN: llc < %s -march=x86-64 | FileCheck %s

define <2 x double> @ld(<2 x double> %p) nounwind optsize ssp {
; CHECK: unpcklpd
  %shuffle = shufflevector <2 x double> %p, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %shuffle
}

define <2 x double> @hd(<2 x double> %p) nounwind optsize ssp {
; CHECK: unpckhpd
  %shuffle = shufflevector <2 x double> %p, <2 x double> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x double> %shuffle
}

define <2 x i64> @ldi(<2 x i64> %p) nounwind optsize ssp {
; CHECK: punpcklqdq
  %shuffle = shufflevector <2 x i64> %p, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %shuffle
}

define <2 x i64> @hdi(<2 x i64> %p) nounwind optsize ssp {
; CHECK: punpckhqdq
  %shuffle = shufflevector <2 x i64> %p, <2 x i64> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x i64> %shuffle
}


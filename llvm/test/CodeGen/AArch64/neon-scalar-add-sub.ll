; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define <1 x i64> @add1xi64(<1 x i64> %A, <1 x i64> %B) {
;CHECK: add {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
	%tmp3 = add <1 x i64> %A, %B;
	ret <1 x i64> %tmp3
}

define <1 x i64> @sub1xi64(<1 x i64> %A, <1 x i64> %B) {
;CHECK: sub {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
	%tmp3 = sub <1 x i64> %A, %B;
	ret <1 x i64> %tmp3
}

declare <1 x i64> @llvm.aarch64.neon.vaddds(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.aarch64.neon.vadddu(<1 x i64>, <1 x i64>)

define <1 x i64> @test_add_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_add_v1i64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vaddds(<1 x i64> %lhs, <1 x i64> %rhs)
; CHECK: add {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}

define <1 x i64> @test_uadd_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_uadd_v1i64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vadddu(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: add {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}

declare <1 x i64> @llvm.aarch64.neon.vsubds(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.aarch64.neon.vsubdu(<1 x i64>, <1 x i64>)

define <1 x i64> @test_sub_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_sub_v1i64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vsubds(<1 x i64> %lhs, <1 x i64> %rhs)
; CHECK: sub {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}

define <1 x i64> @test_usub_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_usub_v1i64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vsubdu(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: sub {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}




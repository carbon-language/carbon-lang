; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s
; Radar 8407927: Make sure that VMOVRRD gets optimized away when the result is
; converted back to be used as a vector type.

; CHECK: test:
define <4 x i32> @test() nounwind {
entry:
  br i1 undef, label %bb1, label %bb2

bb1:
  %0 = bitcast <2 x i64> zeroinitializer to <2 x double>
  %1 = extractelement <2 x double> %0, i32 0
  %2 = bitcast double %1 to i64
  %3 = insertelement <1 x i64> undef, i64 %2, i32 0
; CHECK-NOT: vmov s
; CHECK: vext.8
  %4 = shufflevector <1 x i64> %3, <1 x i64> undef, <2 x i32> <i32 0, i32 1>
  %tmp2006.3 = bitcast <2 x i64> %4 to <16 x i8>
  %5 = shufflevector <16 x i8> %tmp2006.3, <16 x i8> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
  %tmp2004.3 = bitcast <16 x i8> %5 to <4 x i32>
  br i1 undef, label %bb2, label %bb1

bb2:
  %result = phi <4 x i32> [ undef, %entry ], [ %tmp2004.3, %bb1 ]
  ret <4 x i32> %result
}

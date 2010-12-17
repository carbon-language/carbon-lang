; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

; PR7158
define arm_aapcs_vfpcc i32 @test_pr7158() nounwind {
bb.nph55.bb.nph55.split_crit_edge:
  br label %bb3

bb3:                                              ; preds = %bb3, %bb.nph55.bb.nph55.split_crit_edge
  br i1 undef, label %bb.i19, label %bb3

bb.i19:                                           ; preds = %bb.i19, %bb3
  %0 = insertelement <4 x float> undef, float undef, i32 3 ; <<4 x float>> [#uses=3]
  %1 = fmul <4 x float> %0, %0                    ; <<4 x float>> [#uses=1]
  %2 = bitcast <4 x float> %1 to <2 x double>     ; <<2 x double>> [#uses=0]
  %3 = fmul <4 x float> %0, undef                 ; <<4 x float>> [#uses=0]
  br label %bb.i19
}

; Check that the DAG combiner does not arbitrarily modify BUILD_VECTORs
; after legalization.
define void @test_illegal_build_vector() nounwind {
entry:
  store <2 x i64> undef, <2 x i64>* undef, align 16
  %0 = load <16 x i8>* undef, align 16            ; <<16 x i8>> [#uses=1]
  %1 = or <16 x i8> zeroinitializer, %0           ; <<16 x i8>> [#uses=1]
  store <16 x i8> %1, <16 x i8>* undef, align 16
  ret void
}

; Radar 8407927: Make sure that VMOVRRD gets optimized away when the result is
; converted back to be used as a vector type.
; CHECK: test_vmovrrd_combine
define <4 x i32> @test_vmovrrd_combine() nounwind {
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

; Test trying to do a ShiftCombine on illegal types.
; The vector should be split first.
define void @lshrIllegalType(<8 x i32>* %A) nounwind {
       %tmp1 = load <8 x i32>* %A
       %tmp2 = lshr <8 x i32> %tmp1, < i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
       store <8 x i32> %tmp2, <8 x i32>* %A
       ret void
}


; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -disable-fp-elim                -arm-vdup-splat | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -relocation-model=pic -disable-fp-elim -arm-vdup-splat | FileCheck %s 
; Modified version of machine-licm.ll with -arm-vdup-splat turned on, 8003375.
; Eventually this should become the default and be moved into machine-licm.ll.
; FIXME: the vdup should be hoisted out of the loop, 8248029.

define void @t2(i8* %ptr1, i8* %ptr2) nounwind {
entry:
; CHECK: t2:
; CHECK: mov.w r3, #1065353216
  br i1 undef, label %bb1, label %bb2

bb1:
; CHECK-NEXT: %bb1
; CHECK: vdup.32 q1, r3
  %indvar = phi i32 [ %indvar.next, %bb1 ], [ 0, %entry ]
  %tmp1 = shl i32 %indvar, 2
  %gep1 = getelementptr i8* %ptr1, i32 %tmp1
  %tmp2 = call <4 x float> @llvm.arm.neon.vld1.v4f32(i8* %gep1, i32 1)
  %tmp3 = call <4 x float> @llvm.arm.neon.vmaxs.v4f32(<4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float> %tmp2)
  %gep2 = getelementptr i8* %ptr2, i32 %tmp1
  call void @llvm.arm.neon.vst1.v4f32(i8* %gep2, <4 x float> %tmp3, i32 1)
  %indvar.next = add i32 %indvar, 1
  %cond = icmp eq i32 %indvar.next, 10
  br i1 %cond, label %bb2, label %bb1

bb2:
  ret void
}

; CHECK-NOT: LCPI1_0:
; CHECK: .subsections_via_symbols

declare <4 x float> @llvm.arm.neon.vld1.v4f32(i8*, i32) nounwind readonly

declare void @llvm.arm.neon.vst1.v4f32(i8*, <4 x float>, i32) nounwind

declare <4 x float> @llvm.arm.neon.vmaxs.v4f32(<4 x float>, <4 x float>) nounwind readnone

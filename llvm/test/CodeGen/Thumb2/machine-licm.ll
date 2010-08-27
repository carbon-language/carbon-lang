; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -disable-fp-elim                       | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -relocation-model=pic -disable-fp-elim | FileCheck %s --check-prefix=PIC
; rdar://7353541
; rdar://7354376

; The generated code is no where near ideal. It's not recognizing the two
; constantpool entries being loaded can be merged into one.

@GV = external global i32                         ; <i32*> [#uses=2]

define void @t1(i32* nocapture %vals, i32 %c) nounwind {
entry:
; CHECK: t1:
; CHECK: cbz
  %0 = icmp eq i32 %c, 0                          ; <i1> [#uses=1]
  br i1 %0, label %return, label %bb.nph

bb.nph:                                           ; preds = %entry
; CHECK: BB#1
; CHECK: ldr.n r2, LCPI0_0
; CHECK: ldr r2, [r2]
; CHECK: ldr r3, [r2]
; CHECK: LBB0_2
; CHECK: LCPI0_0:
; CHECK-NOT: LCPI0_1:

; PIC: BB#1
; PIC: ldr.n r2, LCPI0_0
; PIC: add r2, pc
; PIC: ldr r2, [r2]
; PIC: ldr r3, [r2]
; PIC: LBB0_2
; PIC: LCPI0_0:
; PIC-NOT: LCPI0_1:
; PIC: .section
  %.pre = load i32* @GV, align 4                  ; <i32> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
  %1 = phi i32 [ %.pre, %bb.nph ], [ %3, %bb ]    ; <i32> [#uses=1]
  %i.03 = phi i32 [ 0, %bb.nph ], [ %4, %bb ]     ; <i32> [#uses=2]
  %scevgep = getelementptr i32* %vals, i32 %i.03  ; <i32*> [#uses=1]
  %2 = load i32* %scevgep, align 4                ; <i32> [#uses=1]
  %3 = add nsw i32 %1, %2                         ; <i32> [#uses=2]
  store i32 %3, i32* @GV, align 4
  %4 = add i32 %i.03, 1                           ; <i32> [#uses=2]
  %exitcond = icmp eq i32 %4, %c                  ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret void
}

; rdar://8001136
define void @t2(i8* %ptr1, i8* %ptr2) nounwind {
entry:
; CHECK: t2:
; CHECK: adr r{{.}}, #LCPI1_0
; CHECK: vldmia r3, {d0, d1}
  br i1 undef, label %bb1, label %bb2

bb1:
; CHECK-NEXT: %bb1
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

; CHECK: LCPI1_0:
; CHECK: .section

declare <4 x float> @llvm.arm.neon.vld1.v4f32(i8*, i32) nounwind readonly

declare void @llvm.arm.neon.vst1.v4f32(i8*, <4 x float>, i32) nounwind

declare <4 x float> @llvm.arm.neon.vmaxs.v4f32(<4 x float>, <4 x float>) nounwind readnone

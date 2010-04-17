; RUN: llc < %s -mtriple=thumbv7-apple-darwin -disable-fp-elim                       | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -relocation-model=pic -disable-fp-elim | FileCheck %s --check-prefix=PIC
; rdar://7353541
; rdar://7354376

; The generated code is no where near ideal. It's not recognizing the two
; constantpool entries being loaded can be merged into one.

@GV = external global i32                         ; <i32*> [#uses=2]

define arm_apcscc void @t(i32* nocapture %vals, i32 %c) nounwind {
entry:
; CHECK: t:
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
; CHECK-NOT: LCPI1_1:
; CHECK: .section

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

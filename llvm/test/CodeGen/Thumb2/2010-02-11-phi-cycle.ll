; RUN: llc < %s -mtriple=thumbv7-apple-darwin | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"

define arm_apcscc i32 @test(i32 %n) nounwind {
; CHECK: test:
; CHECK-NOT: mov
; CHECK: return
entry:
  %0 = icmp eq i32 %n, 1                          ; <i1> [#uses=1]
  br i1 %0, label %return, label %bb.nph

bb.nph:                                           ; preds = %entry
  %tmp = add i32 %n, -1                           ; <i32> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb.nph, %bb
  %indvar = phi i32 [ 0, %bb.nph ], [ %indvar.next, %bb ] ; <i32> [#uses=1]
  %u.05 = phi i64 [ undef, %bb.nph ], [ %ins, %bb ] ; <i64> [#uses=1]
  %1 = tail call arm_apcscc  i32 @f() nounwind    ; <i32> [#uses=1]
  %tmp4 = zext i32 %1 to i64                      ; <i64> [#uses=1]
  %mask = and i64 %u.05, -4294967296              ; <i64> [#uses=1]
  %ins = or i64 %tmp4, %mask                      ; <i64> [#uses=2]
  tail call arm_apcscc  void @g(i64 %ins) nounwind
  %indvar.next = add i32 %indvar, 1               ; <i32> [#uses=2]
  %exitcond = icmp eq i32 %indvar.next, %tmp      ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret i32 undef
}

declare arm_apcscc i32 @f()

declare arm_apcscc void @g(i64)

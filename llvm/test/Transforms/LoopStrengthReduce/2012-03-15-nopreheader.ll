; RUN: opt -loop-reduce -S < %s | FileCheck %s
;
; <rdar://problem/11049788> Segmentation fault: 11 in LoopStrengthReduce

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; IVUsers should not consider tmp128 a valid user because it is not in a
; simplified loop nest.
; CHECK: @nopreheader
; CHECK: for.cond:
; CHECK: %tmp128 = add i64 %0, %indvar65
define void @nopreheader(i8* %cmd) nounwind ssp {
entry:
  indirectbr i8* undef, [label %while.cond]

while.cond:                                       ; preds = %while.body, %entry
  %0 = phi i64 [ %indvar.next48, %while.body ], [ 0, %entry ]
  indirectbr i8* undef, [label %while.end, label %while.body]

while.body:                                       ; preds = %lor.rhs, %lor.lhs.false17, %lor.lhs.false11, %lor.lhs.false, %land.rhs
  %indvar.next48 = add i64 %0, 1
  indirectbr i8* undef, [label %while.cond]

while.end:                                        ; preds = %lor.rhs, %while.cond
  indirectbr i8* undef, [label %if.end152]

if.end152:                                        ; preds = %lor.lhs.false144, %if.end110
  indirectbr i8* undef, [label %lor.lhs.false184, label %for.cond]

lor.lhs.false184:                                 ; preds = %lor.lhs.false177
  indirectbr i8* undef, [label %return, label %for.cond]

for.cond:                                         ; preds = %for.inc, %lor.lhs.false184, %if.end152
  %indvar65 = phi i64 [ %indvar.next66, %for.inc ], [ 0, %lor.lhs.false184 ], [ 0, %if.end152 ]
  %tmp128 = add i64 %0, %indvar65
  %s.4 = getelementptr i8* %cmd, i64 %tmp128
  %tmp195 = load i8* %s.4, align 1
  indirectbr i8* undef, [label %return, label %land.rhs198]

land.rhs198:                                      ; preds = %for.cond
  indirectbr i8* undef, [label %return, label %for.inc]

for.inc:                                          ; preds = %lor.rhs234, %land.lhs.true228, %land.lhs.true216, %land.lhs.true204
  %indvar.next66 = add i64 %indvar65, 1
  indirectbr i8* undef, [label %for.cond]

return:                                           ; preds = %if.end677, %doshell, %if.then96
  ret void
}

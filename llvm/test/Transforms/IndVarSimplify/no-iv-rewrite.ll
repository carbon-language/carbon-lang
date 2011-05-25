; RUN: opt < %s -indvars -disable-iv-rewrite -S | FileCheck %s
;
; Make sure that indvars isn't inserting canonical IVs.
; This is kinda hard to do until linear function test replacement is removed.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define i32 @sum(i32* %arr, i32 %n) nounwind {
entry:
  %precond = icmp slt i32 0, %n
  br i1 %precond, label %ph, label %return

ph:
  br label %loop

; CHECK: loop:
;
; We should only have 2 IVs.
; CHECK: phi
; CHECK: phi
; CHECK-NOT: phi
;
; sext should be eliminated while preserving gep inboundsness.
; CHECK-NOT: sext
; CHECK: getelementptr inbounds
loop:
  %i.02 = phi i32 [ 0, %ph ], [ %iinc, %loop ]
  %s.01 = phi i32 [ 0, %ph ], [ %sinc, %loop ]
  %ofs = sext i32 %i.02 to i64
  %adr = getelementptr inbounds i32* %arr, i64 %ofs
  %val = load i32* %adr
  %sinc = add nsw i32 %s.01, %val
  %iinc = add nsw i32 %i.02, 1
  %cond = icmp slt i32 %iinc, %n
  br i1 %cond, label %loop, label %exit

exit:
  %s.lcssa = phi i32 [ %sinc, %loop ]
  br label %return

return:
  %s.0.lcssa = phi i32 [ %s.lcssa, %exit ], [ 0, %entry ]
  ret i32 %s.0.lcssa
}

define i64 @suml(i32* %arr, i32 %n) nounwind {
entry:
  %precond = icmp slt i32 0, %n
  br i1 %precond, label %ph, label %return

ph:
  br label %loop

; CHECK: loop:
;
; We should only have 2 IVs.
; CHECK: phi
; CHECK: phi
; CHECK-NOT: phi
;
; %ofs sext should be eliminated while preserving gep inboundsness.
; CHECK-NOT: sext
; CHECK: getelementptr inbounds
; %vall sext should obviously not be eliminated
; CHECK: sext
loop:
  %i.02 = phi i32 [ 0, %ph ], [ %iinc, %loop ]
  %s.01 = phi i64 [ 0, %ph ], [ %sinc, %loop ]
  %ofs = sext i32 %i.02 to i64
  %adr = getelementptr inbounds i32* %arr, i64 %ofs
  %val = load i32* %adr
  %vall = sext i32 %val to i64
  %sinc = add nsw i64 %s.01, %vall
  %iinc = add nsw i32 %i.02, 1
  %cond = icmp slt i32 %iinc, %n
  br i1 %cond, label %loop, label %exit

exit:
  %s.lcssa = phi i64 [ %sinc, %loop ]
  br label %return

return:
  %s.0.lcssa = phi i64 [ %s.lcssa, %exit ], [ 0, %entry ]
  ret i64 %s.0.lcssa
}

define void @outofbounds(i32* %first, i32* %last, i32 %idx) nounwind {
  %precond = icmp ne i32* %first, %last
  br i1 %precond, label %ph, label %return

; CHECK: ph:
; It's not indvars' job to perform LICM on %ofs
; CHECK-NOT: sext
ph:
  br label %loop

; CHECK: loop:
;
; Preserve exactly one pointer type IV.
; CHECK: phi i32*
; CHECK-NOT: phi
;
; Don't create any extra adds.
; CHECK-NOT: add
;
; Preserve gep inboundsness, and don't factor it.
; CHECK: getelementptr inbounds i32* %ptriv, i32 1
; CHECK-NOT: add
loop:
  %ptriv = phi i32* [ %first, %ph ], [ %ptrpost, %loop ]
  %ofs = sext i32 %idx to i64
  %adr = getelementptr inbounds i32* %ptriv, i64 %ofs
  store i32 3, i32* %adr
  %ptrpost = getelementptr inbounds i32* %ptriv, i32 1
  %cond = icmp ne i32* %ptrpost, %last
  br i1 %cond, label %loop, label %exit

exit:
  br label %return

return:
  ret void
}

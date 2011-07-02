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
; CHECK: exit:
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
; CHECK: exit:
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
; CHECK: exit:
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

%struct = type { i32 }

define void @bitcastiv(i32 %start, i32 %limit, i32 %step, %struct* %base)
nounwind
{
entry:
  br label %loop

; CHECK: loop:
;
; Preserve casts
; CHECK: phi i32
; CHECK: bitcast
; CHECK: getelementptr
; CHECK: exit:
loop:
  %iv = phi i32 [%start, %entry], [%next, %loop]
  %p = phi %struct* [%base, %entry], [%pinc, %loop]
  %adr = getelementptr %struct* %p, i32 0, i32 0
  store i32 3, i32* %adr
  %pp = bitcast %struct* %p to i32*
  store i32 4, i32* %pp
  %pinc = getelementptr %struct* %p, i32 1
  %next = add i32 %iv, 1
  %cond = icmp ne i32 %next, %limit
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

define void @maxvisitor(i32 %limit, i32* %base) nounwind {
entry:
 br label %loop

; Test inserting a truncate at a phi use.
;
; CHECK: loop:
; CHECK: phi i64
; CHECK: trunc
; CHECK: exit:
loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %loop.inc ]
  %max = phi i32 [ 0, %entry ], [ %max.next, %loop.inc ]
  %idxprom = sext i32 %idx to i64
  %adr = getelementptr inbounds i32* %base, i64 %idxprom
  %val = load i32* %adr
  %cmp19 = icmp sgt i32 %val, %max
  br i1 %cmp19, label %if.then, label %if.else

if.then:
  br label %loop.inc

if.else:
  br label %loop.inc

loop.inc:
  %max.next = phi i32 [ %idx, %if.then ], [ %max, %if.else ]
  %idx.next = add nsw i32 %idx, 1
  %cmp = icmp slt i32 %idx.next, %limit
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @identityphi(i32 %limit) nounwind {
entry:
  br label %loop

; Test an edge case of removing an identity phi that directly feeds
; back to the loop iv.
;
; CHECK: loop:
; CHECK: phi i32
; CHECK-NOT: phi
; CHECK: exit:
loop:
  %iv = phi i32 [ 0, %entry], [ %iv.next, %control ]
  br i1 undef, label %if.then, label %control

if.then:
  br label %control

control:
  %iv.next = phi i32 [ %iv, %loop ], [ undef, %if.then ]
  %cmp = icmp slt i32 %iv.next, %limit
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define i64 @cloneOr(i32 %limit, i64* %base) nounwind {
entry:
  ; ensure that the loop can't overflow
  %halfLim = ashr i32 %limit, 2
  br label %loop

; Test cloning an or, which is not an OverflowBinaryOperator.
;
; CHECK: loop:
; CHECK: phi i64
; CHECK-NOT: sext
; CHECK: or i64
; CHECK: exit:
loop:
  %iv = phi i32 [ 0, %entry], [ %iv.next, %loop ]
  %t1 = sext i32 %iv to i64
  %adr = getelementptr i64* %base, i64 %t1
  %val = load i64* %adr
  %t2 = or i32 %iv, 1
  %t3 = sext i32 %t2 to i64
  %iv.next = add i32 %iv, 2
  %cmp = icmp slt i32 %iv.next, %halfLim
  br i1 %cmp, label %loop, label %exit

exit:
  %result = and i64 %val, %t3
  ret i64 %result
}

; The i induction variable looks like a wrap-around, but it really is just
; a simple affine IV.  Make sure that indvars simplifies through.
define i32 @indirectRecurrence() nounwind {
entry:
  br label %loop

; ReplaceLoopExitValue should fold the return value to constant 9.
; CHECK: loop:
; CHECK: phi i32
; CHECK: ret i32 9
loop:
  %j.0 = phi i32 [ 1, %entry ], [ %j.next, %cond_true ]
  %i.0 = phi i32 [ 0, %entry ], [ %j.0, %cond_true ]
  %tmp = icmp ne i32 %j.0, 10
  br i1 %tmp, label %cond_true, label %return

cond_true:
  %j.next = add i32 %j.0, 1
  br label %loop

return:
  ret i32 %i.0
}

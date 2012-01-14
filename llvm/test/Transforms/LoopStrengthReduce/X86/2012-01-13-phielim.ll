; RUN: llc < %s -O3 -march=x86-64 -mcpu=core2 | FileCheck %s

declare i1 @check() nounwind
declare i1 @foo(i8*, i8*, i8*) nounwind

; Check that redundant phi elimination ran
; CHECK: @test
; CHECK: %while.body.i
; CHECK: movs
; CHECK-NOT: movs
; CHECK: %for.end.i
define i32 @test(i8* %base) nounwind uwtable ssp {
entry:
  br label %while.body.lr.ph.i

while.body.lr.ph.i:                               ; preds = %cond.true.i
  br label %while.body.i

while.body.i:                                     ; preds = %cond.true29.i, %while.body.lr.ph.i
  %indvars.iv7.i = phi i64 [ 16, %while.body.lr.ph.i ], [ %indvars.iv.next8.i, %cond.true29.i ]
  %i.05.i = phi i64 [ 0, %while.body.lr.ph.i ], [ %indvars.iv7.i, %cond.true29.i ]
  %sext.i = shl i64 %i.05.i, 32
  %idx.ext.i = ashr exact i64 %sext.i, 32
  %add.ptr.sum.i = add i64 %idx.ext.i, 16
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %while.body.i
  %indvars.iv.i = phi i64 [ 0, %while.body.i ], [ %indvars.iv.next.i, %for.body.i ]
  %add.ptr.sum = add i64 %add.ptr.sum.i, %indvars.iv.i
  %arrayidx22.i = getelementptr inbounds i8* %base, i64 %add.ptr.sum
  %0 = load i8* %arrayidx22.i, align 1
  %indvars.iv.next.i = add i64 %indvars.iv.i, 1
  %cmp = call i1 @check() nounwind
  br i1 %cmp, label %for.end.i, label %for.body.i

for.end.i:                                        ; preds = %for.body.i
  %add.ptr.i144 = getelementptr inbounds i8* %base, i64 %add.ptr.sum.i
  %cmp2 = tail call i1 @foo(i8* %add.ptr.i144, i8* %add.ptr.i144, i8* undef) nounwind
  br i1 %cmp2, label %cond.true29.i, label %cond.false35.i

cond.true29.i:                                    ; preds = %for.end.i
  %indvars.iv.next8.i = add i64 %indvars.iv7.i, 16
  br i1 false, label %exit, label %while.body.i

cond.false35.i:                                   ; preds = %for.end.i
  unreachable

exit:                                 ; preds = %cond.true29.i, %cond.true.i
  ret i32 0
}

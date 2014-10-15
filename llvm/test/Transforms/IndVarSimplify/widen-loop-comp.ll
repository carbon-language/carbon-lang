; RUN: opt < %s -indvars -S | FileCheck %s
target triple = "aarch64--linux-gnu"

; Check the loop exit i32 compare instruction and operand are widened to i64
; instead of truncating IV before its use in the i32 compare instruction.

@idx = common global i32 0, align 4
@e = common global i32 0, align 4
@ptr = common global i32* null, align 8

; CHECK-LABEL: @test1
; CHECK: for.body.lr.ph:
; CHECK: sext i32
; CHECK: for.cond:
; CHECK: icmp slt i64
; CHECK: for.body:
; CHECK: phi i64

define i32 @test1() {
entry:
  store i32 -1, i32* @idx, align 4
  %0 = load i32* @e, align 4
  %cmp4 = icmp slt i32 %0, 0
  br i1 %cmp4, label %for.end.loopexit, label %for.body.lr.ph

for.body.lr.ph:
  %1 = load i32** @ptr, align 8
  %2 = load i32* @e, align 4
  br label %for.body

for.cond:
  %inc = add nsw i32 %i.05, 1
  %cmp = icmp slt i32 %i.05, %2
  br i1 %cmp, label %for.body, label %for.cond.for.end.loopexit_crit_edge

for.body:
  %i.05 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.cond ]
  %idxprom = sext i32 %i.05 to i64
  %arrayidx = getelementptr inbounds i32* %1, i64 %idxprom
  %3 = load i32* %arrayidx, align 4
  %tobool = icmp eq i32 %3, 0
  br i1 %tobool, label %if.then, label %for.cond

if.then:
  %i.05.lcssa = phi i32 [ %i.05, %for.body ]
  store i32 %i.05.lcssa, i32* @idx, align 4
  br label %for.end

for.cond.for.end.loopexit_crit_edge:
  br label %for.end.loopexit

for.end.loopexit:
  br label %for.end

for.end:
  %4 = load i32* @idx, align 4
  ret i32 %4
}

; CHECK-LABEL: @test2
; CHECK: for.body4.us
; CHECK: %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK: %cmp2.us = icmp slt i64
; CHECK-NOT: %2 = trunc i64 %indvars.iv.next to i32
; CHECK-NOT: %cmp2.us = icmp slt i32

define void @test2([8 x i8]* %a, i8* %b, i8 %limit) {
entry:
  %conv = zext i8 %limit to i32
  %cmp23 = icmp eq i8 %limit, 0
  br i1 %cmp23, label %for.cond1.preheader, label %for.cond1.preheader.us

for.cond1.preheader.us:
  %storemerge5.us = phi i32 [ 0, %entry ], [ %inc14.us, %for.inc13.us ]
  br i1 true, label %for.body4.lr.ph.us, label %for.inc13.us

for.inc13.us:
  %inc14.us = add nsw i32 %storemerge5.us, 1
  %cmp.us = icmp slt i32 %inc14.us, 4
  br i1 %cmp.us, label %for.cond1.preheader.us, label %for.end

for.body4.us:
  %storemerge14.us = phi i32 [ 0, %for.body4.lr.ph.us ], [ %inc.us, %for.body4.us ]
  %idxprom.us = sext i32 %storemerge14.us to i64
  %arrayidx6.us = getelementptr inbounds [8 x i8]* %a, i64 %idxprom5.us, i64 %idxprom.us
  %0 = load i8* %arrayidx6.us, align 1
  %idxprom7.us = zext i8 %0 to i64
  %arrayidx8.us = getelementptr inbounds i8* %b, i64 %idxprom7.us
  %1 = load i8* %arrayidx8.us, align 1
  store i8 %1, i8* %arrayidx6.us, align 1
  %inc.us = add nsw i32 %storemerge14.us, 1
  %cmp2.us = icmp slt i32 %inc.us, %conv
  br i1 %cmp2.us, label %for.body4.us, label %for.inc13.us

for.body4.lr.ph.us:
  %idxprom5.us = sext i32 %storemerge5.us to i64
  br label %for.body4.us

for.cond1.preheader:
  %storemerge5 = phi i32 [ 0, %entry ], [ %inc14, %for.inc13 ]
  br i1 false, label %for.inc13, label %for.inc13

for.inc13:
  %inc14 = add nsw i32 %storemerge5, 1
  %cmp = icmp slt i32 %inc14, 4
  br i1 %cmp, label %for.cond1.preheader, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @test3
; CHECK: sext i32 %b
; CHECK: for.cond:
; CHECK: phi i64
; CHECK: icmp slt i64

define i32 @test3(i32* %a, i32 %b) {
entry:
  br label %for.cond

for.cond:
  %sum.0 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, %b
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32* %a, i64 %idxprom
  %0 = load i32* %arrayidx, align 4
  %add = add nsw i32 %sum.0, %0
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret i32 %sum.0
}

declare i32 @fn1(i8 signext)

; PR21030
; CHECK-LABEL: @test4
; CHECK: for.body:
; CHECK: phi i32
; CHECK: icmp sgt i8

define i32 @test4(i32 %a) {
entry:
  br label %for.body

for.body:
  %c.07 = phi i8 [ -3, %entry ], [ %dec, %for.body ]
  %conv6 = zext i8 %c.07 to i32
  %or = or i32 %a, %conv6
  %conv3 = trunc i32 %or to i8
  %call = call i32 @fn1(i8 signext %conv3)
  %dec = add i8 %c.07, -1
  %cmp = icmp sgt i8 %dec, -14
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret i32 0
}

; CHECK-LABEL: @test5
; CHECK: zext i32 %b
; CHECK: for.cond:
; CHECK: phi i64
; CHECK: icmp ule i64

define i32 @test5(i32* %a, i32 %b) {
entry:
  br label %for.cond

for.cond:
  %sum.0 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp ule i32 %i.0, %b
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32* %a, i64 %idxprom
  %0 = load i32* %arrayidx, align 4
  %add = add nsw i32 %sum.0, %0
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret i32 %sum.0
}

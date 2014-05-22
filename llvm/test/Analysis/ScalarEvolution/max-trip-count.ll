; RUN: opt < %s -analyze -scalar-evolution | FileCheck %s

; ScalarEvolution should be able to understand the loop and eliminate the casts.

; CHECK: {%d,+,sizeof(i32)}

define void @foo(i32* nocapture %d, i32 %n) nounwind {
entry:
	%0 = icmp sgt i32 %n, 0		; <i1> [#uses=1]
	br i1 %0, label %bb.nph, label %return

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb1, %bb.nph
	%i.02 = phi i32 [ %5, %bb1 ], [ 0, %bb.nph ]		; <i32> [#uses=2]
	%p.01 = phi i8 [ %4, %bb1 ], [ -1, %bb.nph ]		; <i8> [#uses=2]
	%1 = sext i8 %p.01 to i32		; <i32> [#uses=1]
	%2 = sext i32 %i.02 to i64		; <i64> [#uses=1]
	%3 = getelementptr i32* %d, i64 %2		; <i32*> [#uses=1]
	store i32 %1, i32* %3, align 4
	%4 = add i8 %p.01, 1		; <i8> [#uses=1]
	%5 = add i32 %i.02, 1		; <i32> [#uses=2]
	br label %bb1

bb1:		; preds = %bb
	%6 = icmp slt i32 %5, %n		; <i1> [#uses=1]
	br i1 %6, label %bb, label %bb1.return_crit_edge

bb1.return_crit_edge:		; preds = %bb1
	br label %return

return:		; preds = %bb1.return_crit_edge, %entry
	ret void
}

; ScalarEvolution should be able to find the maximum tripcount
; of this multiple-exit loop, and if it doesn't know the exact
; count, it should say so.

; PR7845
; CHECK: Loop %for.cond: <multiple exits> Unpredictable backedge-taken count. 
; CHECK: Loop %for.cond: max backedge-taken count is 5

@.str = private constant [4 x i8] c"%d\0A\00"     ; <[4 x i8]*> [#uses=2]

define i32 @main() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %g_4.0 = phi i32 [ 0, %entry ], [ %add, %for.inc ] ; <i32> [#uses=5]
  %cmp = icmp slt i32 %g_4.0, 5                   ; <i1> [#uses=1]
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %conv = trunc i32 %g_4.0 to i16                 ; <i16> [#uses=1]
  %tobool.not = icmp eq i16 %conv, 0              ; <i1> [#uses=1]
  %tobool3 = icmp ne i32 %g_4.0, 0                ; <i1> [#uses=1]
  %or.cond = and i1 %tobool.not, %tobool3         ; <i1> [#uses=1]
  br i1 %or.cond, label %for.end, label %for.inc

for.inc:                                          ; preds = %for.body
  %add = add nsw i32 %g_4.0, 1                    ; <i32> [#uses=1]
  br label %for.cond

for.end:                                          ; preds = %for.body, %for.cond
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i32 %g_4.0) nounwind ; <i32> [#uses=0]
  ret i32 0
}

declare i32 @printf(i8*, ...)

define void @test(i8* %a, i32 %n) nounwind {
entry:
  %cmp1 = icmp sgt i32 %n, 0
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %tmp = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %for.body.lr.ph ]
  %arrayidx = getelementptr i8* %a, i64 %indvar
  store i8 0, i8* %arrayidx, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar.next, %tmp
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void
}

; CHECK: Determining loop execution counts for: @test
; CHECK-NEXT: backedge-taken count is
; CHECK-NEXT: max backedge-taken count is -1

; PR19799: Indvars miscompile due to an incorrect max backedge taken count from SCEV.
; CHECK-LABEL: @pr19799
; CHECK: Loop %for.body.i: <multiple exits> Unpredictable backedge-taken count. 
; CHECK: Loop %for.body.i: max backedge-taken count is 1
@a = common global i32 0, align 4

define i32 @pr19799() {
entry:
  store i32 -1, i32* @a, align 4
  br label %for.body.i

for.body.i:                                       ; preds = %for.cond.i, %entry
  %storemerge1.i = phi i32 [ -1, %entry ], [ %add.i.i, %for.cond.i ]
  %tobool.i = icmp eq i32 %storemerge1.i, 0
  %add.i.i = add nsw i32 %storemerge1.i, 2
  br i1 %tobool.i, label %bar.exit, label %for.cond.i

for.cond.i:                                       ; preds = %for.body.i
  store i32 %add.i.i, i32* @a, align 4
  %cmp.i = icmp slt i32 %storemerge1.i, 0
  br i1 %cmp.i, label %for.body.i, label %bar.exit

bar.exit:                                         ; preds = %for.cond.i, %for.body.i
  ret i32 0
}

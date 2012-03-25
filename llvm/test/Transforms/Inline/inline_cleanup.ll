; Test that the inliner doesn't leave around dead allocas, and that it folds
; uncond branches away after it is done specializing.

; RUN: opt < %s -inline -S | FileCheck %s

@A = weak global i32 0		; <i32*> [#uses=1]
@B = weak global i32 0		; <i32*> [#uses=1]
@C = weak global i32 0		; <i32*> [#uses=1]

define internal fastcc void @foo(i32 %X) {
entry:
	%ALL = alloca i32, align 4		; <i32*> [#uses=1]
	%tmp1 = and i32 %X, 1		; <i32> [#uses=1]
	%tmp1.upgrd.1 = icmp eq i32 %tmp1, 0		; <i1> [#uses=1]
	br i1 %tmp1.upgrd.1, label %cond_next, label %cond_true

cond_true:		; preds = %entry
	store i32 1, i32* @A
	br label %cond_next

cond_next:		; preds = %cond_true, %entry
	%tmp4 = and i32 %X, 2		; <i32> [#uses=1]
	%tmp4.upgrd.2 = icmp eq i32 %tmp4, 0		; <i1> [#uses=1]
	br i1 %tmp4.upgrd.2, label %cond_next7, label %cond_true5

cond_true5:		; preds = %cond_next
	store i32 1, i32* @B
	br label %cond_next7

cond_next7:		; preds = %cond_true5, %cond_next
	%tmp10 = and i32 %X, 4		; <i32> [#uses=1]
	%tmp10.upgrd.3 = icmp eq i32 %tmp10, 0		; <i1> [#uses=1]
	br i1 %tmp10.upgrd.3, label %cond_next13, label %cond_true11

cond_true11:		; preds = %cond_next7
	store i32 1, i32* @C
	br label %cond_next13

cond_next13:		; preds = %cond_true11, %cond_next7
	%tmp16 = and i32 %X, 8		; <i32> [#uses=1]
	%tmp16.upgrd.4 = icmp eq i32 %tmp16, 0		; <i1> [#uses=1]
	br i1 %tmp16.upgrd.4, label %UnifiedReturnBlock, label %cond_true17

cond_true17:		; preds = %cond_next13
	call void @ext( i32* %ALL )
	ret void

UnifiedReturnBlock:		; preds = %cond_next13
	ret void
}

declare void @ext(i32*)

define void @test() {
; CHECK: @test
; CHECK-NOT: ret
;
; FIXME: This should be a CHECK-NOT, but currently we have a bug that causes us
; to not nuke unused allocas.
; CHECK: alloca
; CHECK-NOT: ret
;
; No branches should survive the inliner's cleanup.
; CHECK-NOT: br
; CHECK: ret void

entry:
	tail call fastcc void @foo( i32 1 )
	tail call fastcc void @foo( i32 2 )
	tail call fastcc void @foo( i32 3 )
	tail call fastcc void @foo( i32 8 )
	ret void
}

declare void @f(i32 %x)

define void @inner2(i32 %x, i32 %y, i32 %z, i1 %b) {
entry:
  %cmp1 = icmp ne i32 %x, 0
  br i1 %cmp1, label %then1, label %end1

then1:
  call void @f(i32 %x)
  br label %end1

end1:
  %x2 = and i32 %x, %z
  %cmp2 = icmp sgt i32 %x2, 1
  br i1 %cmp2, label %then2, label %end2

then2:
  call void @f(i32 %x2)
  br label %end2

end2:
  %y2 = or i32 %y, %z
  %cmp3 = icmp sgt i32 %y2, 0
  br i1 %cmp3, label %then3, label %end3

then3:
  call void @f(i32 %y2)
  br label %end3

end3:
  br i1 %b, label %end3.1, label %end3.2

end3.1:
  %x3.1 = or i32 %x, 10
  br label %end3.3

end3.2:
  %x3.2 = or i32 %x, 10
  br label %end3.3

end3.3:
  %x3.3 = phi i32 [ %x3.1, %end3.1 ], [ %x3.2, %end3.2 ]
  %cmp4 = icmp slt i32 %x3.3, 1
  br i1 %cmp4, label %then4, label %end4

then4:
  call void @f(i32 %x3.3)
  br label %end4

end4:
  ret void
}

define void @outer2(i32 %z, i1 %b) {
; Ensure that after inlining, none of the blocks with a call to @f actually
; make it through inlining.
; CHECK: define void @outer2
; CHECK-NOT: call
;
; FIXME: Currently, we aren't smart enough to delete the last dead basic block.
; However, we do make the condition a constant. Check that at least until we can
; start removing the block itself.
; CHECK: br i1 false, label %[[LABEL:[a-z0-9_.]+]],
; CHECK-NOT: call
; CHECK: [[LABEL]]:
; CHECK-NEXT: call void @f(i32 10)
; CHECK-NOT: call
;
; CHECK: ret void

entry:
  call void @inner2(i32 0, i32 -1, i32 %z, i1 %b)
  ret void
}

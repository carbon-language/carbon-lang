; RUN: opt < %s -basicaa -gvn -S | FileCheck %s

@a = external global i32		; <i32*> [#uses=7]

; CHECK: @test1
define i32 @test1() nounwind {
entry:
	%0 = load i32* @a, align 4
	%1 = icmp eq i32 %0, 4
	br i1 %1, label %bb, label %bb1

bb:		; preds = %entry
	br label %bb8

bb1:		; preds = %entry
	%2 = load i32* @a, align 4
	%3 = icmp eq i32 %2, 5
	br i1 %3, label %bb2, label %bb3

bb2:		; preds = %bb1
	br label %bb8

bb3:		; preds = %bb1
	%4 = load i32* @a, align 4
	%5 = icmp eq i32 %4, 4
; CHECK: br i1 false, label %bb4, label %bb5
	br i1 %5, label %bb4, label %bb5

bb4:		; preds = %bb3
	%6 = load i32* @a, align 4
	%7 = add i32 %6, 5
	br label %bb8

bb5:		; preds = %bb3
	%8 = load i32* @a, align 4
	%9 = icmp eq i32 %8, 5
; CHECK: br i1 false, label %bb6, label %bb7
	br i1 %9, label %bb6, label %bb7

bb6:		; preds = %bb5
	%10 = load i32* @a, align 4
	%11 = add i32 %10, 4
	br label %bb8

bb7:		; preds = %bb5
	%12 = load i32* @a, align 4
	br label %bb8

bb8:		; preds = %bb7, %bb6, %bb4, %bb2, %bb
	%.0 = phi i32 [ %12, %bb7 ], [ %11, %bb6 ], [ %7, %bb4 ], [ 4, %bb2 ], [ 5, %bb ]
	br label %return

return:		; preds = %bb8
	ret i32 %.0
}

declare void @foo(i1)

; CHECK: @test2
define void @test2(i1 %x, i1 %y) {
  %z = or i1 %x, %y
  br i1 %z, label %true, label %false
true:
; CHECK: true:
  %z2 = or i1 %x, %y
  call void @foo(i1 %z2)
; CHECK: call void @foo(i1 true)
  br label %true
false:
; CHECK: false:
  %z3 = or i1 %x, %y
  call void @foo(i1 %z3)
; CHECK: call void @foo(i1 false)
  br label %false
}

declare void @bar(i32)

; CHECK: @test3
define void @test3(i32 %x, i32 %y) {
  %xz = icmp eq i32 %x, 0
  %yz = icmp eq i32 %y, 0
  %z = and i1 %xz, %yz
  br i1 %z, label %both_zero, label %nope
both_zero:
  call void @foo(i1 %xz)
; CHECK: call void @foo(i1 true)
  call void @foo(i1 %yz)
; CHECK: call void @foo(i1 true)
  call void @bar(i32 %x)
; CHECK: call void @bar(i32 0)
  call void @bar(i32 %y)
; CHECK: call void @bar(i32 0)
  ret void
nope:
  call void @foo(i1 %z)
; CHECK: call void @foo(i1 false)
  ret void
}

; RUN: opt < %s -jump-threading -S | FileCheck %s

declare i32 @f1()
declare i32 @f2()
declare void @f3()

define i32 @test1(i1 %cond) {
; CHECK: @test1

	br i1 %cond, label %T1, label %F1

T1:
	%v1 = call i32 @f1()
	br label %Merge

F1:
	%v2 = call i32 @f2()
	br label %Merge

Merge:
	%A = phi i1 [true, %T1], [false, %F1]
	%B = phi i32 [%v1, %T1], [%v2, %F1]
	br i1 %A, label %T2, label %F2

T2:
; CHECK: T2:
; CHECK: ret i32 %v1
	call void @f3()
	ret i32 %B

F2:
; CHECK: F2:
; CHECK: ret i32 %v2
	ret i32 %B
}


;; cond is known false on Entry -> F1 edge!
define i32 @test2(i1 %cond) {
; CHECK: @test2
Entry:
	br i1 %cond, label %T1, label %F1

T1:
; CHECK: %v1 = call i32 @f1()
; CHECK: ret i32 47
	%v1 = call i32 @f1()
	br label %Merge

F1:
	br i1 %cond, label %Merge, label %F2

Merge:
	%B = phi i32 [47, %T1], [192, %F1]
	ret i32 %B

F2:
	call void @f3()
	ret i32 12
}


; Undef handling.
define i32 @test3(i1 %cond) {
; CHECK: @test3
; CHECK-NEXT: T1:
; CHECK-NEXT: ret i32 42
	br i1 undef, label %T1, label %F1

T1:
	ret i32 42

F1:
	ret i32 17
}

define i32 @test4(i1 %cond, i1 %cond2) {
; CHECK: @test4

	br i1 %cond, label %T1, label %F1

T1:
; CHECK:   %v1 = call i32 @f1()
; CHECK-NEXT:   br label %T

	%v1 = call i32 @f1()
	br label %Merge

F1:
	%v2 = call i32 @f2()
; CHECK:   %v2 = call i32 @f2()
; CHECK-NEXT:   br i1 %cond2, 
	br label %Merge

Merge:
	%A = phi i1 [undef, %T1], [%cond2, %F1]
	%B = phi i32 [%v1, %T1], [%v2, %F1]
	br i1 %A, label %T2, label %F2

T2:
	call void @f3()
	ret i32 %B

F2:
	ret i32 %B
}


;; This tests that the branch in 'merge' can be cloned up into T1.
define i32 @test5(i1 %cond, i1 %cond2) {
; CHECK: @test5

	br i1 %cond, label %T1, label %F1

T1:
; CHECK: T1:
; CHECK-NEXT:   %v1 = call i32 @f1()
; CHECK-NEXT:   %cond3 = icmp eq i32 %v1, 412
; CHECK-NEXT:   br i1 %cond3, label %T2, label %F2

	%v1 = call i32 @f1()
        %cond3 = icmp eq i32 %v1, 412
	br label %Merge

F1:
	%v2 = call i32 @f2()
	br label %Merge

Merge:
	%A = phi i1 [%cond3, %T1], [%cond2, %F1]
	%B = phi i32 [%v1, %T1], [%v2, %F1]
	br i1 %A, label %T2, label %F2

T2:
	call void @f3()
	ret i32 %B

F2:
	ret i32 %B
}


;; Lexically duplicated conditionals should be threaded.


define i32 @test6(i32 %A) {
; CHECK: @test6
	%tmp455 = icmp eq i32 %A, 42
	br i1 %tmp455, label %BB1, label %BB2
        
BB2:
; CHECK: call i32 @f1()
; CHECK-NEXT: call void @f3()
; CHECK-NEXT: ret i32 4
	call i32 @f1()
	br label %BB1
        

BB1:
	%tmp459 = icmp eq i32 %A, 42
	br i1 %tmp459, label %BB3, label %BB4

BB3:
	call i32 @f2()
        ret i32 3

BB4:
	call void @f3()
	ret i32 4
}





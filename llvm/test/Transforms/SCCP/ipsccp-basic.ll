; RUN: opt < %s -ipsccp -S | FileCheck %s
; XFAIL: *

;;======================== test1

define internal i32 @test1a(i32 %A) {
	%X = add i32 1, 2
	ret i32 %A
}
; CHECK: define internal i32 @test1a
; CHECK: ret i32 undef

define i32 @test1b() {
	%X = call i32 @test1a( i32 17 )
	ret i32 %X

; CHECK: define i32 @test1b
; CHECK: ret i32 17
}



;;======================== test2

define internal i32 @test2a(i32 %A) {
	%C = icmp eq i32 %A, 0	
	br i1 %C, label %T, label %F
T:
	%B = call i32 @test2a( i32 0 )
	ret i32 0
F:
	%C.upgrd.1 = call i32 @test2a(i32 1)
	ret i32 %C.upgrd.1
}
; CHECK: define internal i32 @test2a
; CHECK-NEXT: br label %T
; CHECK: ret i32 undef


define i32 @test2b() {
	%X = call i32 @test2a(i32 0)
	ret i32 %X
}
; CHECK: define i32 @test2b
; CHECK-NEXT: %X = call i32 @test2a(i32 0)
; CHECK-NEXT: ret i32 0


;;======================== test3

@G = internal global i32 undef

define void @test3a() {
	%X = load i32* @G
	store i32 %X, i32* @G
	ret void
}
; CHECK: define void @test3a
; CHECK-NEXT: ret void


define i32 @test3b() {
	%V = load i32* @G
	%C = icmp eq i32 %V, 17
	br i1 %C, label %T, label %F
T:
	store i32 17, i32* @G
	ret i32 %V
F:	
	store i32 123, i32* @G
	ret i32 0
}
; CHECK: define i32 @test3b
; CHECK-NOT: store
; CHECK: ret i32 0


;;======================== test4

define internal {i64,i64} @test4a() {
  %a = insertvalue {i64,i64} undef, i64 4, 1
  %b = insertvalue {i64,i64} %a, i64 5, 0
  ret {i64,i64} %b
}

define i64 @test4b() {
  %a = invoke {i64,i64} @test4a()
          to label %A unwind label %B
A:
  %b = extractvalue {i64,i64} %a, 0
  %c = call i64 @test4c(i64 %b)
  ret i64 %c
B:
  ret i64 0
}
; CHECK: define i64 @test4b()
; CHECK:   %c = call i64 @test4c(i64 5)
; CHECK-NEXT:  ret i64 5


define internal i64 @test4c(i64 %a) {
  ret i64 %a
}
; CHECK: define internal i64 @test4c
; CHECK: ret i64 undef



;;======================== test5

; PR4313
define internal {i64,i64} @test5a() {
  %a = insertvalue {i64,i64} undef, i64 4, 1
  %b = insertvalue {i64,i64} %a, i64 5, 0
  ret {i64,i64} %b
}

define i64 @test5b() {
  %a = invoke {i64,i64} @test5a()
          to label %A unwind label %B
A:
  %c = call i64 @test5c({i64,i64} %a)
  ret i64 %c
B:
  ret i64 0
}

; CHECK: define i64 @test5b()
; CHECK:     A:
; CHECK-NEXT:  %c = call i64 @test5c(%0 %a)
; CHECK-NEXT:  ret i64 %c

define internal i64 @test5c({i64,i64} %a) {
  %b = extractvalue {i64,i64} %a, 0
  ret i64 %b
}


;;======================== test6

define i64 @test6a() {
  ret i64 0
}

define i64 @test6b() {
  %a = call i64 @test6a()
  ret i64 %a
}
; CHECK: define i64 @test6b
; CHECK: ret i64 0

;;======================== test7


%T = type {i32,i32}

define internal {i32, i32} @test7a(i32 %A) {
  %X = add i32 1, %A
  %mrv0 = insertvalue %T undef, i32 %X, 0
  %mrv1 = insertvalue %T %mrv0, i32 %A, 1
  ret %T %mrv1
; CHECK: @test7a
; CHECK-NEXT: %mrv0 = insertvalue %T undef, i32 18, 0
; CHECK-NEXT: %mrv1 = insertvalue %T %mrv0, i32 17, 1
}

define i32 @test7b() {
	%X = call {i32, i32} @test7a(i32 17)
        %Y = extractvalue {i32, i32} %X, 0
	%Z = add i32 %Y, %Y
	ret i32 %Z
; CHECK: define i32 @test7b
; CHECK-NEXT: call %T @test7a(i32 17)
; CHECK-NEXT: ret i32 36
}



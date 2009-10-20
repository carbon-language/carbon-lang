; RUN: opt < %s -ipsccp -S | FileCheck %s

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

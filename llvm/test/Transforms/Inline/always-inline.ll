; RUN: opt < %s -inline-threshold=0 -always-inline -S | FileCheck %s
;
; Ensure the threshold has no impact on these decisions.
; RUN: opt < %s -inline-threshold=20000000 -always-inline -S | FileCheck %s
; RUN: opt < %s -inline-threshold=-20000000 -always-inline -S | FileCheck %s

define i32 @inner1() alwaysinline {
  ret i32 1
}
define i32 @outer1() {
; CHECK: @outer1
; CHECK-NOT: call
; CHECK: ret

   %r = call i32 @inner1()
   ret i32 %r
}

; The always inliner can't DCE internal functions. PR2945
; CHECK: @pr2945
define internal i32 @pr2945() nounwind {
  ret i32 0
}

define internal void @inner2(i32 %N) alwaysinline {
  %P = alloca i32, i32 %N
  ret void
}
define void @outer2(i32 %N) {
; The always inliner (unlike the normal one) should be willing to inline
; a function with a dynamic alloca into one without a dynamic alloca.
; rdar://6655932
;
; CHECK: @outer2
; CHECK-NOT: call void @inner2
; CHECK alloca i32, i32 %N
; CHECK-NOT: call void @inner2
; CHECK: ret void

  call void @inner2( i32 %N )
  ret void
}

declare i32 @a() returns_twice
declare i32 @b() returns_twice

define i32 @inner3() alwaysinline {
entry:
  %call = call i32 @a() returns_twice
  %add = add nsw i32 1, %call
  ret i32 %add
}
define i32 @outer3() {
entry:
; CHECK: @outer3
; CHECK-NOT: call i32 @a
; CHECK: ret

  %call = call i32 @inner3()
  %add = add nsw i32 1, %call
  ret i32 %add
}

define i32 @inner4() alwaysinline returns_twice {
entry:
  %call = call i32 @b() returns_twice
  %add = add nsw i32 1, %call
  ret i32 %add
}

define i32 @outer4() {
entry:
; CHECK: @outer4
; CHECK: call i32 @b()
; CHECK: ret

  %call = call i32 @inner4() returns_twice
  %add = add nsw i32 1, %call
  ret i32 %add
}

; RUN: opt -S -mergefunc < %s | FileCheck %s

; Minor note: functions need to be at least three instructions long
; to be considered by -mergefunc.

declare i32 @foo(...)

define i32 @f() {
; CHECK-LABEL: @f(
 entry:
  %v0 = call i32 (...) @foo(i32 10) [ "foo"(i32 20) ]
  %v1 = call i32 (...) @foo(i32 10) [ "foo"(i32 20) ]
  %v2 = call i32 (...) @foo(i32 10) [ "foo"(i32 20) ]

; CHECK:  %v0 = call i32 (...) @foo(i32 10) [ "foo"(i32 20) ]
; CHECK:  %v1 = call i32 (...) @foo(i32 10) [ "foo"(i32 20) ]
; CHECK:  %v2 = call i32 (...) @foo(i32 10) [ "foo"(i32 20) ]

  ret i32 %v2
}

define i32 @g() {
; CHECK-LABEL: @g(
 entry:
  %v0 = call i32 (...) @foo() [ "foo"(i32 10, i32 20) ]
  %v1 = call i32 (...) @foo() [ "foo"(i32 10, i32 20) ]
  %v2 = call i32 (...) @foo() [ "foo"(i32 10, i32 20) ]

; CHECK:  %v0 = call i32 (...) @foo() [ "foo"(i32 10, i32 20) ]
; CHECK:  %v1 = call i32 (...) @foo() [ "foo"(i32 10, i32 20) ]
; CHECK:  %v2 = call i32 (...) @foo() [ "foo"(i32 10, i32 20) ]

  ret i32 %v2
}

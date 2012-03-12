; RUN: opt < %s -inline -S | FileCheck %s

define internal i32 @callee1(i32 %A, i32 %B) {
  %C = sdiv i32 %A, %B
  ret i32 %C
}

define i32 @caller1() {
; CHECK: define i32 @caller1
; CHECK-NEXT: ret i32 3

  %X = call i32 @callee1( i32 10, i32 3 )
  ret i32 %X
}

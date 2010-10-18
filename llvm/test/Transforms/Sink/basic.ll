; RUN: opt < %s -basicaa -sink -S | FileCheck %s

@A = external global i32
@B = external global i32

; Sink should sink the load past the store (which doesn't overlap) into
; the block that uses it.

;      CHECK: @foo
;      CHECK: true:
; CHECK-NEXT: %l = load i32* @A
; CHECK-NEXT: ret i32 %l

define i32 @foo(i1 %z) {
  %l = load i32* @A
  store i32 0, i32* @B
  br i1 %z, label %true, label %false
true:
  ret i32 %l
false:
  ret i32 0
}

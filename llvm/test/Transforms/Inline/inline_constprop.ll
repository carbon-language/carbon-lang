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

define i32 @caller2() {
; CHECK: @caller2
; CHECK-NOT: call void @callee2
; CHECK: ret

; We contrive to make this hard for *just* the inline pass to do in order to
; simulate what can actually happen with large, complex functions getting
; inlined.
  %a = add i32 42, 0
  %b = add i32 48, 0

  %x = call i32 @callee21(i32 %a, i32 %b)
  ret i32 %x
}

define i32 @callee21(i32 %x, i32 %y) {
  %sub = sub i32 %y, %x
  %result = call i32 @callee22(i32 %sub)
  ret i32 %result
}

declare i8* @getptr()

define i32 @callee22(i32 %x) {
  %icmp = icmp ugt i32 %x, 42
  br i1 %icmp, label %bb.true, label %bb.false
bb.true:
  ; This block musn't be counted in the inline cost.
  %ptr = call i8* @getptr()
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr
  load volatile i8* %ptr

  ret i32 %x
bb.false:
  ret i32 %x
}

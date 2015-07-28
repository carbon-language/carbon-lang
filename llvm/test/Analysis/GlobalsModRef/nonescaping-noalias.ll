; RUN: opt < %s -globalsmodref-aa -gvn -S | FileCheck %s
;
; This tests the safe no-alias conclusions of GMR -- when there is
; a non-escaping global as one indentified underlying object and some pointer
; that would inherently have escaped any other function as the other underlying
; pointer of an alias query.

@g1 = internal global i32 0

define i32 @test1(i32* %param) {
; Ensure that we can fold a store to a load of a global across a store to
; a parameter when the global is non-escaping.
;
; CHECK-LABEL: @test1(
; CHECK: store i32 42, i32* @g1
; CHECK-NOT: load i32
; CHECK: ret i32 42
entry:
  store i32 42, i32* @g1
  store i32 7, i32* %param
  %v = load i32, i32* @g1
  ret i32 %v
}

declare i32* @f()

define i32 @test2() {
; Ensure that we can fold a store to a load of a global across a store to
; the pointer returned by a function call. Since the global could not escape,
; this function cannot be returning its address.
;
; CHECK-LABEL: @test2(
; CHECK: store i32 42, i32* @g1
; CHECK-NOT: load i32
; CHECK: ret i32 42
entry:
  %ptr = call i32* @f() readnone
  store i32 42, i32* @g1
  store i32 7, i32* %ptr
  %v = load i32, i32* @g1
  ret i32 %v
}

@g2 = external global i32*

define i32 @test3() {
; Ensure that we can fold a store to a load of a global across a store to
; the pointer loaded from that global. Because the global does not escape, it
; cannot alias a pointer loaded out of a global.
;
; CHECK-LABEL: @test3(
; CHECK: store i32 42, i32* @g1
; CHECK: store i32 7, i32*
; CHECK-NOT: load i32
; CHECK: ret i32 42
entry:
  store i32 42, i32* @g1
  %ptr1 = load i32*, i32** @g2
  store i32 7, i32* %ptr1
  %v = load i32, i32* @g1
  ret i32 %v
}

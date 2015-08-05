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

define i32 @test4(i32* %param, i32 %n, i1 %c1, i1 %c2) {
; Ensure that we can fold a store to a load of a global across a store to
; the pointer loaded from that global even when the load is behind PHIs and
; selects, and there is a mixture of a load and another global or argument.
; Note that we can't eliminate the load here because it is used in a PHI and
; GVN doesn't try to do real DCE. The store is still forwarded by GVN though.
;
; CHECK-LABEL: @test4(
; CHECK: store i32 42, i32* @g1
; CHECK: store i32 7, i32*
; CHECK: ret i32 42
entry:
  %call = call i32* @f()
  store i32 42, i32* @g1
  %ptr1 = load i32*, i32** @g2
  %ptr2 = select i1 %c1, i32* %ptr1, i32* %param
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %ptr = phi i32* [ %ptr2, %entry ], [ %ptr4, %loop ]
  store i32 7, i32* %ptr
  %ptr3 = load i32*, i32** @g2
  %ptr4 = select i1 %c2, i32* %ptr3, i32* %call
  %inc = add i32 %iv, 1
  %test = icmp slt i32 %inc, %n
  br i1 %test, label %loop, label %exit

exit:
  %v = load i32, i32* @g1
  ret i32 %v
}

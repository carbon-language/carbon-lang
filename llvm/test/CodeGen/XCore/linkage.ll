; RUN: llc < %s -march=xcore | FileCheck %s

; CHECK: .weak fd
define weak void @fd() {
  call void @fr(i32* @gd, i32* @gr)
  ret void
}

; CHECK-NOT: .hidden
define hidden void @test_hidden() {
  unreachable
}

; CHECK-NOT: .protected
define protected void @test_protected() {
  unreachable
}

; CHECK: .globl array.globound
; CHECK: array.globound = 2
; CHECK: .weak array.globound
; CHECK: .globl array
; CHECK: .weak array
@array = weak global [2 x i32] zeroinitializer

; CHECK: .weak gd
@gd = weak global i32 0

; CHECK: .weak gr
@gr = extern_weak global i32

; CHECK: .weak fr
declare extern_weak void @fr(i32*, i32*)


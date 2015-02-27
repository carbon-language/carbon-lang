; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s
; Test that we don't crashe if the .Leh_func_end0 name is taken.

declare void @g()

define void @f() {
bb0:
  call void asm ".Leh_func_end0:", ""()
; CHECK: #APP
; CHECK-NEXT: .Leh_func_end0:
; CHECK-NEXT: #NO_APP

  invoke void @g() to label %bb2 unwind label %bb1
bb1:
  landingpad { i8*, i32 } personality i8* bitcast (void ()* @g to i8*)
          catch i8* null
  call void @g()
  ret void
bb2:
  ret void

; CHECK: [[END:.Leh_func_end.*]]:
; CHECK: .long	[[END]]-
}

; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s

; Test that we can handle .Lexception0 being defined. We used to crash.

; CHECK: .cfi_lsda 3, [[LABEL:.*]]
; CHECK: [[LABEL]]:
; CHECK-NEXT: .byte   255                     # @LPStart Encoding = omit

declare void @g()

define void @f() {
bb0:
  call void asm ".Lexception0:", ""()
  invoke void @g()
          to label %bb2 unwind label %bb1
bb1:
  landingpad { i8*, i32 } personality i8* bitcast (void ()* @g to i8*)
          catch i8* null
  br label %bb2

bb2:
  ret void
}

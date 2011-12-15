; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck  %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare i32 @clock() nounwind

define i32 @func() {
entry:
  %call = call i32 @clock() nounwind
  %call2 = add i32 %call, 7
  ret i32 %call2
}

; CHECK: bl clock
; CHECK-NEXT: nop


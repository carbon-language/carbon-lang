; RUN: opt < %s -tailcallelim -S | FileCheck %s

define void @t() nounwind ssp {
entry:
; CHECK: entry:
; CHECK: %0 = call i32 @foo()
; CHECK: ret void
  %0 = call i32 @foo() nounwind noreturn
  ret void
}

declare i32 @foo()

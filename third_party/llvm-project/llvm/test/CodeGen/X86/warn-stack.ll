; RUN: llc -mtriple x86_64-apple-macosx10.8.0 < %s 2>&1 >/dev/null | FileCheck %s
; Check the internal option that warns when the stack frame size exceeds the
; given amount.
; <rdar://13987214>

; CHECK-NOT: nowarn
define void @nowarn() nounwind ssp "warn-stack-size"="80" {
entry:
  %buffer = alloca [12 x i8], align 1
  %arraydecay = getelementptr inbounds [12 x i8], [12 x i8]* %buffer, i64 0, i64 0
  call void @doit(i8* %arraydecay) nounwind
  ret void
}

; CHECK: warning: stack frame size (88) exceeds limit (80) in function 'warn'
define void @warn() nounwind ssp "warn-stack-size"="80" {
entry:
  %buffer = alloca [80 x i8], align 1
  %arraydecay = getelementptr inbounds [80 x i8], [80 x i8]* %buffer, i64 0, i64 0
  call void @doit(i8* %arraydecay) nounwind
  ret void
}

; Ensure that warn-stack-size also considers the size of the unsafe stack.
; With safestack enabled the machine stack size is well below 80, but the
; combined stack size of the machine stack and unsafe stack will exceed the
; warning threshold

; CHECK: warning: stack frame size (120) exceeds limit (80) in function 'warn_safestack'
define void @warn_safestack() nounwind ssp safestack "warn-stack-size"="80" {
entry:
  %buffer = alloca [80 x i8], align 1
  %arraydecay = getelementptr inbounds [80 x i8], [80 x i8]* %buffer, i64 0, i64 0
  call void @doit(i8* %arraydecay) nounwind
  ret void
}
declare void @doit(i8*)

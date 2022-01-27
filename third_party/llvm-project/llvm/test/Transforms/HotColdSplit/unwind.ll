; RUN: opt -hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; Do not split out `resume` instructions.

; CHECK-LABEL: define {{.*}}@foo.cold.1(
; CHECK: call {{.*}}@sink(
; CHECK-NOT: resume i32 undef

; CHECK-NOT: noreturn

define i32 @foo() personality i8 0 {
entry:
  invoke void @llvm.donothing() to label %normal unwind label %exception

exception:
  %cleanup = landingpad i32 cleanup
  br i1 undef, label %normal, label %continue_exception

continue_exception:
  call void @sideeffect(i32 0)
  call void @sink()
  br label %resume-eh

resume-eh:
  resume i32 undef

normal:
  br i1 undef, label %continue_exception, label %exit

exit:
  call void @sideeffect(i32 2)
  ret i32 0
}

declare void @sideeffect(i32)

declare void @sink() cold

declare void @llvm.donothing() nounwind readnone

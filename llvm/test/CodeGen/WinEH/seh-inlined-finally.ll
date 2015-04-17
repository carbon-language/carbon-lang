; RUN: opt -S -winehprepare -sehprepare < %s | FileCheck %s

; Check that things work when the mid-level optimizer inlines the finally
; block.

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare i32 @puts(i8*)
declare void @may_crash()
declare i32 @__C_specific_handler(...)

define void @use_finally() {
entry:
  invoke void @may_crash()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %call.i = tail call i32 @puts(i8* null)
  ret void

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
          cleanup
  %call.i2 = tail call i32 @puts(i8* null)
  resume { i8*, i32 } %0
}

; CHECK-LABEL: define void @use_finally()
; CHECK: invoke void @may_crash()
;
; CHECK: landingpad
; CHECK-NEXT: cleanup
; CHECK-NEXT: call i8* (...) @llvm.eh.actions(i32 0, void (i8*, i8*)* @use_finally.cleanup)
; CHECK-NEXT: indirectbr i8* %recover, []

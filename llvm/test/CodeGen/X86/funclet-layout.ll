; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @f(i1 %B) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @g()
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:
  %cp = catchpad [i8* null, i32 64, i8* null]
          to label %catch unwind label %catchendblock

catch:
  br i1 %B, label %catchret, label %catch

catchret:
  catchret %cp to label %try.cont

try.cont:
  ret void

catchendblock:
  catchendpad unwind to caller

unreachable:
  unreachable
}

; CHECK-LABEL: f:

; The entry funclet contains %entry and %try.cont
; CHECK: # %entry
; CHECK: # %try.cont
; CHECK: retq

; The catch funclet contains %catch and %catchret
; CHECK: # %catch
; CHECK: # %catchret
; CHECK: retq

declare void @g()

declare i32 @__CxxFrameHandler3(...)

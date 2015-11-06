; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

declare i32 @__CxxFrameHandler3(...)

define i8* @f() personality i32 (...)* @__CxxFrameHandler3 {
bb:
  unreachable

unreachable:
  %cl = cleanuppad []
  cleanupret %cl unwind to caller
}

; CHECK: unreachable:
; CHECK:   %cl = cleanuppad []
; CHECK:   cleanupret %cl unwind to caller


declare void @g(i8*)

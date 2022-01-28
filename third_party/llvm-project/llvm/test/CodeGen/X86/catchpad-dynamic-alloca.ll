; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @rt_init()

declare i32 @__CxxFrameHandler3(...)

define void @test1(void ()* %fp, i64 %n) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %t.i = alloca i8*
  %t.ii = alloca i8
  %.alloca8 = alloca i8, i64 %n
  store volatile i8 0, i8* %t.ii
  store volatile i8 0, i8* %.alloca8
  invoke void @rt_init()
          to label %try.cont unwind label %catch.switch

try.cont:
  invoke void %fp()
          to label %exit unwind label %catch.switch

exit:
  ret void

catch.pad:
  %cp = catchpad within %cs [i8* null, i32 0, i8** %t.i]
  catchret from %cp to label %exit

catch.switch:
  %cs = catchswitch within none [label %catch.pad] unwind to caller
}

; CHECK-LABEL: $handlerMap$0$test1:
; CHECK:      .long   0
; CHECK-NEXT: .long   0
; CHECK-NEXT: .long   16

define void @test2(void ()* %fp, i64 %n) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %t.i = alloca i128
  %.alloca8 = alloca i8, i64 %n
  store volatile i8 0, i8* %.alloca8
  invoke void @rt_init()
          to label %try.cont unwind label %catch.switch

try.cont:
  invoke void %fp()
          to label %exit unwind label %catch.switch

exit:
  ret void

catch.pad:
  %cp = catchpad within %cs [i8* null, i32 0, i128* %t.i]
  catchret from %cp to label %exit

catch.switch:
  %cs = catchswitch within none [label %catch.pad] unwind to caller
}

; CHECK-LABEL: $handlerMap$0$test2:
; CHECK:      .long   0
; CHECK-NEXT: .long   0
; CHECK-NEXT: .long   8

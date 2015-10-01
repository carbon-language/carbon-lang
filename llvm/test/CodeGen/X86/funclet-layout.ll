; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%eh.ThrowInfo = type { i32, i32, i32, i32 }

define void @test1(i1 %B) personality i32 (...)* @__CxxFrameHandler3 {
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

; CHECK-LABEL: test1:

; The entry funclet contains %entry and %try.cont
; CHECK: # %entry
; CHECK: # %try.cont
; CHECK: retq

; The catch funclet contains %catch and %catchret
; CHECK: # %catch
; CHECK: # %catchret
; CHECK: retq

declare void @g()


define i32 @test2(i1 %B) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null) #1
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchpad [i8* null, i32 64, i8* null]
          to label %catch unwind label %catchendblock

catch:                                            ; preds = %catch.dispatch
  invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null) #1
          to label %unreachable unwind label %catch.dispatch.1

catch.dispatch.1:                                 ; preds = %catch
  %1 = catchpad [i8* null, i32 64, i8* null]
          to label %catch.3 unwind label %catchendblock.2

catch.3:                                          ; preds = %catch.dispatch.1
  catchret %1 to label %try.cont

try.cont:                                         ; preds = %catch.3
  catchret %0 to label %try.cont.5

try.cont.5:                                       ; preds = %try.cont
  ret i32 0

catchendblock.2:                                  ; preds = %catch.dispatch.1
  catchendpad unwind label %catchendblock

catchendblock:                                    ; preds = %catchendblock.2, %catch.dispatch
  catchendpad unwind to caller

unreachable:                                      ; preds = %catch, %entry
  unreachable

}

; CHECK-LABEL: test2:

; The entry funclet contains %entry and %try.cont.5
; CHECK: # %entry
; CHECK: # %try.cont.5
; CHECK: retq

; The inner catch funclet contains %catch.3
; CHECK: # %catch.3
; CHECK: retq

; The outer catch funclet contains %catch and %try.cont
; CHECK: # %catch
; CHECK: # %try.cont
; CHECK: retq

declare void @_CxxThrowException(i8*, %eh.ThrowInfo*)
declare i32 @__CxxFrameHandler3(...)

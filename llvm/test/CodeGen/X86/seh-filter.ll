; RUN: llc -O0 -mtriple=x86_64-windows-msvc < %s | FileCheck %s

declare void @g()
define void @f() personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
  invoke void @g() to label %return unwind label %lpad

return:
  ret void

lpad:
  %ehptrs = landingpad {i8*, i32}
    filter [0 x i8*] zeroinitializer
  call void @__cxa_call_unexpected(i8* null)
  unreachable
}
declare i32 @__C_specific_handler(...)
declare void @__cxa_call_unexpected(i8*)

; We don't emit entries for filters.
; CHECK: .seh_handlerdata
; CHECK: .long 0

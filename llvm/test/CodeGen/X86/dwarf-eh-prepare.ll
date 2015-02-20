; RUN: opt -mtriple=x86_64-linux-gnu -dwarfehprepare < %s -S | FileCheck %s

; Check basic functionality of IR-to-IR DWARF EH preparation. This should
; eliminate resumes. This pass requires a TargetMachine, so we put it under X86
; and provide an x86 triple.

@int_typeinfo = global i8 0

declare void @might_throw()

define i32 @simple_catch() {
  invoke void @might_throw()
          to label %cont unwind label %lpad

; CHECK: define i32 @simple_catch()
; CHECK: invoke void @might_throw()

cont:
  ret i32 0

; CHECK: ret i32 0

lpad:
  %ehvals = landingpad { i8*, i32 } personality i32 (...)* @__gxx_personality_v0
      catch i8* @int_typeinfo
  %ehptr = extractvalue { i8*, i32 } %ehvals, 0
  %ehsel = extractvalue { i8*, i32 } %ehvals, 1
  %int_sel = call i32 @llvm.eh.typeid.for(i8* @int_typeinfo)
  %int_match = icmp eq i32 %ehsel, %int_sel
  br i1 %int_match, label %catch_int, label %eh.resume

; CHECK: lpad:
; CHECK: landingpad { i8*, i32 } personality i32 (...)* @__gxx_personality_v0
; CHECK: call i32 @llvm.eh.typeid.for
; CHECK: br i1

catch_int:
  ret i32 1

; CHECK: catch_int:
; CHECK: ret i32 1

eh.resume:
  resume { i8*, i32 } %ehvals

; CHECK: eh.resume:
; CHECK: call void @_Unwind_Resume(i8* %{{.*}})
}

declare i32 @__gxx_personality_v0(...)
declare i32 @llvm.eh.typeid.for(i8*)

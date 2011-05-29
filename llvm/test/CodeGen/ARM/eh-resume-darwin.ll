; RUN: llc < %s -march=arm | FileCheck %s
target triple = "armv6-apple-macosx10.6"

declare void @func()

declare i8* @llvm.eh.exception() nounwind readonly

declare i32 @llvm.eh.selector(i8*, i8*, ...) nounwind

declare void @llvm.eh.resume(i8*, i32)

declare i32 @__gxx_personality_sj0(...)

define void @test0() {
entry:
  invoke void @func()
    to label %cont unwind label %lpad

cont:
  ret void

lpad:
  %exn = call i8* @llvm.eh.exception()
  %sel = call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* %exn, i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*), i32 0)
  call void @llvm.eh.resume(i8* %exn, i32 %sel) noreturn
  unreachable
}

; CHECK: __Unwind_SjLj_Resume

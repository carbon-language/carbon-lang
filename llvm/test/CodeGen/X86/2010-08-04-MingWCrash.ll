; RUN: llc < %s -mtriple=i386-pc-mingw32

define void @func() nounwind {
invoke.cont:
  %call = tail call i8* @malloc()
  %a = invoke i32 @bar()
          to label %bb1 unwind label %lpad

bb1:
  ret void

lpad:
  %exn = tail call i8* @llvm.eh.exception() nounwind
  %eh.selector = tail call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* %exn, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1, i8* null) nounwind
  %ehspec.fails = icmp slt i32 %eh.selector, 0
  br i1 %ehspec.fails, label %ehspec.unexpected, label %cleanup

cleanup:
  tail call void @_Unwind_Resume_or_Rethrow(i8* %exn) noreturn nounwind
  unreachable

ehspec.unexpected:
  tail call void @__cxa_call_unexpected(i8* %exn) noreturn nounwind
  unreachable
}

declare noalias i8* @malloc()

declare i8* @llvm.eh.exception() nounwind readonly

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.selector(i8*, i8*, ...) nounwind

declare void @_Unwind_Resume_or_Rethrow(i8*)

declare void @__cxa_call_unexpected(i8*)

declare i32 @bar()

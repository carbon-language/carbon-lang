; RUN: llc -simplifycfg-require-and-preserve-domtree=1 < %s -mtriple=i386-pc-mingw32

define void @func() nounwind personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
invoke.cont:
  %call = tail call i8* @malloc()
  %a = invoke i32 @bar()
          to label %bb1 unwind label %lpad

bb1:
  ret void

lpad:
  %exn.ptr = landingpad { i8*, i32 }
           catch i8* null
  %exn = extractvalue { i8*, i32 } %exn.ptr, 0
  %eh.selector = extractvalue { i8*, i32 } %exn.ptr, 1
  %ehspec.fails = icmp slt i32 %eh.selector, 0
  br i1 %ehspec.fails, label %ehspec.unexpected, label %cleanup

cleanup:
  resume { i8*, i32 } %exn.ptr

ehspec.unexpected:
  tail call void @__cxa_call_unexpected(i8* %exn) noreturn nounwind
  unreachable
}

declare noalias i8* @malloc()

declare i32 @__gxx_personality_v0(...)

declare void @_Unwind_Resume_or_Rethrow(i8*)

declare void @__cxa_call_unexpected(i8*)

declare i32 @bar()

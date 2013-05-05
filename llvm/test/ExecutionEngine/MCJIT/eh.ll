; RUN: %lli_mcjit %s
; XFAIL ppc64
declare i8* @__cxa_allocate_exception(i64)
declare void @__cxa_throw(i8*, i8*, i8*)
declare i32 @__gxx_personality_v0(...)

@_ZTIi = external constant i8*

define void @throwException() {
  %exception = tail call i8* @__cxa_allocate_exception(i64 4)
  call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null)
  unreachable
}

define i32 @main() {
entry:
  invoke void @throwException()
          to label %try.cont unwind label %lpad

lpad:
  %p = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* bitcast (i8** @_ZTIi to i8*)
  br label %try.cont

try.cont:
  ret i32 0
}

; RUN: llc < %s
; PR4317

declare i32 @b()

define void @a() {
entry:
  ret void

dummy:
  invoke i32 @b() to label %reg unwind label %reg

reg:
  %lpad = landingpad { i8*, i32 } personality i32 (...)* @__gxx_personality_v0
            catch i8* null
  ret void
}

declare i32 @__gxx_personality_v0(...)

; RUN: llc < %s
; PR4317

declare i32 @b()

define void @a() {
entry:
  ret void

dummy:
  invoke i32 @b() to label %reg unwind label %reg

reg:
  ret void
}

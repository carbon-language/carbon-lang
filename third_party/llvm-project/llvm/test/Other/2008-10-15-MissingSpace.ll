; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; PR2894
declare void @g()
define void @f() personality i32 (...)* @__gxx_personality_v0 {
; CHECK:  invoke void @g()
; CHECK:           to label %d unwind label %c
  invoke void @g() to label %d unwind label %c
d:
  ret void
c:
  %exn = landingpad {i8*, i32}
            cleanup
  ret void
}

declare i32 @__gxx_personality_v0(...)

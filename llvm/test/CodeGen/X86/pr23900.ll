; RUN: llc -filetype=obj %s -o %t.o
; RUN: llvm-nm %t.o | FileCheck %s

; Test that it doesn't crash (and produces an object file).
; This use to pass a symbol with a null name to code that expected a valid
; C string.

; CHECK:         U __CxxFrameHandler3
; CHECK:         T f
; CHECK:         t f.cleanup
; CHECK:         U g
; CHECK:         U h


target triple = "x86_64-pc-windows-msvc18.0.0"
define void @f(i32 %x) personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
  invoke void @h()
          to label %invoke.cont unwind label %lpad
invoke.cont:
  ret void
lpad:
 landingpad { i8*, i32 }
          cleanup
  call void @g(i32 %x)
  ret void
}
declare void @h()
declare i32 @__CxxFrameHandler3(...)
declare void @g(i32 %x)

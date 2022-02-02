; RUN: llc -mtriple=x86_64-linux < %s | FileCheck %s

; We should treat non-Function personalities as the unknown personality, which
; is usually Itanium.

declare void @g()
declare void @terminate(i8*)

define void @f() personality i8* null {
  invoke void @g()
    to label %ret unwind label %lpad
ret:
  ret void
lpad:
  %vals = landingpad { i8*, i32 } catch i8* null
  %ptr = extractvalue { i8*, i32 } %vals, 0
  call void @terminate(i8* %ptr)
  unreachable
}

; CHECK: f:
; CHECK: callq g
; CHECK: retq
; CHECK: movq %rax, %rdi
; CHECK: callq terminate

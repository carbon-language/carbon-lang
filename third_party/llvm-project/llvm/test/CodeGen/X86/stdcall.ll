; RUN: llc -mtriple="i386-pc-mingw32" < %s | FileCheck %s
; PR5851

%0 = type { void (...)* }

define internal x86_stdcallcc void @MyFunc() nounwind {
entry:
; CHECK: MyFunc@0:
; CHECK: retl
  ret void
}

; PR14410
define x86_stdcallcc i32 @"\01DoNotMangle"(i32 %a) {
; CHECK: DoNotMangle:
; CHECK: retl $4
entry:
  ret i32 %a
}

@B = global %0 { void (...)* bitcast (void ()* @MyFunc to void (...)*) }, align 4
; CHECK: _B:
; CHECK: .long _MyFunc@0


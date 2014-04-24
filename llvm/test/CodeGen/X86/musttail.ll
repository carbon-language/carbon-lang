; RUN: llc -march=x86 < %s | FileCheck %s

; FIXME: Eliminate this tail call at -O0, since musttail is a correctness
; requirement.
; RUN: not llc -march=x86 -O0 < %s

declare void @t1_callee(i8*)
define void @t1(i32* %a) {
; CHECK-LABEL: t1:
; CHECK: jmp {{_?}}t1_callee
  %b = bitcast i32* %a to i8*
  musttail call void @t1_callee(i8* %b)
  ret void
}

declare i8* @t2_callee()
define i32* @t2() {
; CHECK-LABEL: t2:
; CHECK: jmp {{_?}}t2_callee
  %v = musttail call i8* @t2_callee()
  %w = bitcast i8* %v to i32*
  ret i32* %w
}

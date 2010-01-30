; RUN: llc < %s -march=x86    -asm-verbose=false | FileCheck %s -check-prefix=32
; RUN: llc < %s -march=x86-64 -asm-verbose=false | FileCheck %s -check-prefix=64

define void @t1(i32 %x) nounwind ssp {
entry:
; 32: t1:
; 32: jmp {{_?}}foo

; 64: t1:
; 64: jmp {{_?}}foo
  tail call void @foo() nounwind
  ret void
}

declare void @foo()

define void @t2() nounwind ssp {
entry:
; 32: t2:
; 32: jmp {{_?}}foo2

; 64: t2:
; 64: jmp {{_?}}foo2
  %0 = tail call i32 @foo2() nounwind
  ret void
}

declare i32 @foo2()

define void @t3() nounwind ssp {
entry:
; 32: t3:
; 32: jmp {{_?}}foo3

; 64: t3:
; 64: jmp {{_?}}foo3
  %0 = tail call i32 @foo3() nounwind
  ret void
}

declare i32 @foo3()

define void @t4(void (i32)* nocapture %x) nounwind ssp {
entry:
; 32: t4:
; 32: call *
; FIXME: gcc can generate a tailcall for this. But it's tricky.

; 64: t4:
; 64-NOT: call
; 64: jmpq *
  tail call void %x(i32 0) nounwind
  ret void
}

define void @t5(void ()* nocapture %x) nounwind ssp {
entry:
; 32: t5:
; 32-NOT: call
; 32: jmpl *

; 64: t5:
; 64-NOT: call
; 64: jmpq *
  tail call void %x() nounwind
  ret void
}

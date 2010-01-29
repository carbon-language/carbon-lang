; RUN: llc < %s -march=x86    -asm-verbose=false | FileCheck %s
; RUN: llc < %s -march=x86-64 -asm-verbose=false | FileCheck %s

define void @t1(i32 %x) nounwind ssp {
entry:
; CHECK: t1:
; CHECK: jmp {{_?}}foo
  tail call void @foo() nounwind
  ret void
}

declare void @foo()

define void @t2() nounwind ssp {
entry:
; CHECK: t2:
; CHECK: jmp {{_?}}foo2
  %0 = tail call i32 @foo2() nounwind
  ret void
}

declare i32 @foo2()

define void @t3() nounwind ssp {
entry:
; CHECK: t3:
; CHECK: jmp {{_?}}foo3
  %0 = tail call i32 @foo3() nounwind
  ret void
}

declare i32 @foo3()

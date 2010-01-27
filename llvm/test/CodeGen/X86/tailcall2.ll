; RUN: llc < %s -march=x86    -asm-verbose=false | FileCheck %s
; RUN: llc < %s -march=x86-64 -asm-verbose=false | FileCheck %s

define void @bar(i32 %x) nounwind ssp {
entry:
; CHECK: bar:
; CHECK: jmp {{_?}}foo
  tail call void @foo() nounwind
  ret void
}

declare void @foo()

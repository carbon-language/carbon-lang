; RUN: llc < %s -march=x86 | FileCheck -check-prefix=NO-FP %s
; RUN: llc < %s -march=x86 -disable-fp-elim | FileCheck -check-prefix=FP %s
target triple = "i686-apple-darwin8"

define void @func1() noreturn nounwind  {
entry:
; NO-FP: ud2
        unreachable
}

define void @func2() noreturn nounwind  {
entry:
; FP: pushl %ebp
; FP: movl %esp, %ebp
; FP: ud2
        unreachable
}

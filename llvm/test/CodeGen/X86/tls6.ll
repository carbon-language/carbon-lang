; RUN: llvm-as < %s | llc -march=x86 -mtriple=i386-linux-gnu > %t
; RUN: grep {movl	%gs:0, %eax} %t
; RUN: grep {leal	i@NTPOFF(%eax), %eax} %t
; RUN: llvm-as < %s | llc -march=x86-64 -mtriple=x86_64-linux-gnu > %t2
; RUN: grep {movq	%fs:0, %rax} %t2
; RUN: grep {leaq	i@TPOFF(%rax), %rax} %t2

@i = internal thread_local global i32 15

define i32* @f() {
entry:
	ret i32* @i
}

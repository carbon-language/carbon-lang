; RUN: llc < %s -march=x86 -mtriple=i386-linux-gnu > %t
; RUN: grep {movl	%gs:0, %eax} %t
; RUN: grep {addl	i@INDNTPOFF, %eax} %t
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-linux-gnu > %t2
; RUN: grep {movq	%fs:0, %rax} %t2
; RUN: grep {addq	i@GOTTPOFF(%rip), %rax} %t2

@i = external thread_local global i32		; <i32*> [#uses=2]

define i32* @f() {
entry:
	ret i32* @i
}

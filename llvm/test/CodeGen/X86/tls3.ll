; RUN: llc < %s -march=x86 -mtriple=i386-linux-gnu > %t
; RUN: grep {movl	i@INDNTPOFF, %eax} %t
; RUN: grep {movl	%gs:(%eax), %eax} %t
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-linux-gnu > %t2
; RUN: grep {movq	i@GOTTPOFF(%rip), %rax} %t2
; RUN: grep {movl	%fs:(%rax), %eax} %t2

@i = external thread_local global i32		; <i32*> [#uses=2]

define i32 @f() nounwind {
entry:
	%tmp1 = load i32* @i		; <i32> [#uses=1]
	ret i32 %tmp1
}

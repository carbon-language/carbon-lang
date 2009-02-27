; RUN: llvm-as < %s | llc -march=x86 -mtriple=i386-linux-gnu > %t
; RUN: grep {movl	%gs:0, %eax} %t
; RUN: grep {addl	i@INDNTPOFF, %eax} %t

@i = external thread_local global i32		; <i32*> [#uses=2]

define i32* @f() {
entry:
	ret i32* @i
}

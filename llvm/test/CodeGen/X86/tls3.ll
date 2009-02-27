; RUN: llvm-as < %s | llc -march=x86 -mtriple=i386-linux-gnu > %t
; RUN: grep {movl	i@INDNTPOFF, %eax} %t
; RUN: grep {movl	%gs:(%eax), %eax} %t

@i = external thread_local global i32		; <i32*> [#uses=2]

define i32 @f() {
entry:
	%tmp1 = load i32* @i		; <i32> [#uses=1]
	ret i32 %tmp1
}

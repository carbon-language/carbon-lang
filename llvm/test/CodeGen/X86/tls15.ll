; RUN: llvm-as < %s | llc -march=x86 -mtriple=i386-linux-gnu > %t
; RUN: grep {movl	%gs:0, %eax} %t | count 1
; RUN: grep {leal	i@NTPOFF(%eax), %ecx} %t
; RUN: grep {leal	j@NTPOFF(%eax), %eax} %t

@i = thread_local global i32 0
@j = thread_local global i32 0

define void @f(i32** %a, i32** %b) {
entry:
	store i32* @i, i32** %a, align 8
	store i32* @j, i32** %b, align 8
	ret void
}

; RUN: llvm-as < %s | llc -march=x86 -mtriple=i386-linux-gnu > %t
; RUN: grep {movl	%gs:0, %eax} %t
; RUN: grep {leal	i@NTPOFF(%eax), %eax} %t

@i = external hidden thread_local global i32

define i32* @f() {
entry:
	ret i32* @i
}

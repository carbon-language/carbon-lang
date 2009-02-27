; RUN: llvm-as < %s | llc -march=x86 -mtriple=i386-linux-gnu > %t
; RUN: grep {movl	%gs:i@NTPOFF, %eax} %t

@i = internal thread_local global i32 15

define i32 @f() {
entry:
	%tmp1 = load i32* @i
	ret i32 %tmp1
}

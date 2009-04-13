; RUN: llvm-as < %s | llc -march=x86 -mtriple=i386-linux-gnu > %t
; RUN: grep {movl	%gs:i@NTPOFF, %eax} %t
; RUN: llvm-as < %s | llc -march=x86-64 -mtriple=x86_64-linux-gnu > %t2
; RUN: grep {movl	%fs:i@TPOFF, %eax} %t2

@i = thread_local global i32 15

define i32 @f() {
entry:
	%tmp1 = load i32* @i
	ret i32 %tmp1
}

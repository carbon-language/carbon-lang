; RUN: llvm-as < %s | llc -march=x86 -mtriple=i386-linux-gnu > %t
; RUN: grep {movb	%gs:i@NTPOFF, %al} %t
; RUN: llvm-as < %s | llc -march=x86-64 -mtriple=x86_64-linux-gnu > %t2
; RUN: grep {movb	%fs:i@TPOFF, %al} %t2

@i = thread_local global i8 15

define i8 @f() {
entry:
	%tmp1 = load i8* @i
	ret i8 %tmp1
}

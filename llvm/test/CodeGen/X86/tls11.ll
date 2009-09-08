; RUN: llc < %s -march=x86 -mtriple=i386-linux-gnu > %t
; RUN: grep {movw	%gs:i@NTPOFF, %ax} %t
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-linux-gnu > %t2
; RUN: grep {movw	%fs:i@TPOFF, %ax} %t2

@i = thread_local global i16 15

define i16 @f() {
entry:
	%tmp1 = load i16* @i
	ret i16 %tmp1
}

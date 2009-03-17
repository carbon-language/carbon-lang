; RUN: llvm-as < %s | llc -march=x86 -mtriple=i386-linux-gnu > %t
; RUN: grep {movw	%gs:i@NTPOFF, %ax} %t

@i = thread_local global i16 15

define i16 @f() {
entry:
	%tmp1 = load i16* @i
	ret i16 %tmp1
}

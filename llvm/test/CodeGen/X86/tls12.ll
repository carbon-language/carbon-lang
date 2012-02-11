; RUN: llc < %s -march=x86 -mtriple=i386-linux-gnu | FileCheck -check-prefix=X32_LINUX %s
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-linux-gnu | FileCheck -check-prefix=X64_LINUX %s
; RUN: llc < %s -march=x86 -mtriple=x86-pc-win32 | FileCheck -check-prefix=X32_WIN %s
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-pc-win32 | FileCheck -check-prefix=X64_WIN %s

@i = thread_local global i8 15

define i8 @f() {
entry:
	%tmp1 = load i8* @i
	ret i8 %tmp1
}
; X32_LINUX: movb %gs:i@NTPOFF, %al
; X64_LINUX: movb %fs:i@TPOFF, %al
; X32_WIN: movl __tls_index, %eax
; X32_WIN: movl %fs:__tls_array, %ecx
; X32_WIN: movb _i@SECREL(%eax), %al
; X64_WIN: movl _tls_index(%rip), %eax
; X64_WIN: movq %gs:88, %rcx
; X64_WIN: movabsq $i@SECREL, %rcx
; X64_WIN: movb (%rax,%rcx), %al

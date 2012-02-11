; RUN: llc < %s -march=x86 -mtriple=i386-linux-gnu | FileCheck -check-prefix=X32_LINUX %s
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-linux-gnu | FileCheck -check-prefix=X64_LINUX %s
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-pc-win32 | FileCheck -check-prefix=X64_WIN %s
; RUN: llc < %s -march=x86 -mtriple=x86-pc-win32 | FileCheck -check-prefix=X32_WIN %s

@i = internal thread_local global i32 15

define i32 @f() {
entry:
	%tmp1 = load i32* @i
	ret i32 %tmp1
}
; X32_LINUX: movl %gs:i@NTPOFF, %eax
; X64_LINUX: movl %fs:i@TPOFF, %eax
; X32_WIN: movl __tls_index, %eax
; X32_WIN: movl %fs:__tls_array, %ecx
; X32_WIN: movl _i@SECREL(%eax), %eax
; X64_WIN: movl _tls_index(%rip), %eax
; X64_WIN: movabsq $i@SECREL, %rcx

; RUN: llc < %s -march=x86 -mtriple=i386-linux-gnu | FileCheck -check-prefix=X32_LINUX %s
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-linux-gnu | FileCheck -check-prefix=X64_LINUX %s
; RUN: llc < %s -march=x86 -mtriple=x86-pc-win32 | FileCheck -check-prefix=X32_WIN %s
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-pc-win32 | FileCheck -check-prefix=X64_WIN %s

@i = external thread_local global i32		; <i32*> [#uses=2]

define i32* @f() {
entry:
	ret i32* @i
}
; X32_LINUX: movl %gs:0, %eax
; X32_LINUX: addl i@INDNTPOFF, %eax
; X64_LINUX: movq %fs:0, %rax
; X64_LINUX: addq i@GOTTPOFF(%rip), %rax
; X32_WIN: movl __tls_index, %eax
; X32_WIN: movl %fs:__tls_array, %ecx
; X32_WIN: leal _i@SECREL(%eax), %eax
; X64_WIN: movl _tls_index(%rip), %eax
; X64_WIN: movq %gs:88, %rcx
; X64_WIN: addq $i@SECREL, %rax

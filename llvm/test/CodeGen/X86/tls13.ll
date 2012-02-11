; RUN: llc < %s -march=x86 -mtriple=i386-linux-gnu | FileCheck -check-prefix=X32_LINUX %s
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-linux-gnu | FileCheck -check-prefix=X64_LINUX %s
; RUN: llc < %s -march=x86 -mtriple=x86-pc-win32 | FileCheck -check-prefix=X32_WIN %s
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-pc-win32 | FileCheck -check-prefix=X64_WIN %s

@i = thread_local global i16 0
@j = thread_local global i16 0

define void @f() nounwind optsize {
entry:
        %0 = load i16* @i, align 2
        %1 = sext i16 %0 to i32
        tail call void @g(i32 %1) nounwind
        %2 = load i16* @j, align 2
        %3 = zext i16 %2 to i32
        tail call void @h(i32 %3) nounwind
        ret void
}

declare void @g(i32)

declare void @h(i32)

; X32_LINUX: movswl %gs:i@NTPOFF, %eax
; X32_LINUX: movzwl %gs:j@NTPOFF, %eax
; X64_LINUX: movswl %fs:i@TPOFF, %edi
; X64_LINUX: movzwl %fs:j@TPOFF, %edi
; X32_WIN: movswl _i@SECREL(%esi), %eax
; X32_WIN: movzwl _j@SECREL(%esi), %eax
; X64_WIN: movabsq $i@SECREL, %rax
; X64_WIN: movswl (%rsi,%rax), %ecx
; X64_WIN: movabsq $j@SECREL, %rax
; X64_WIN: movzwl (%rsi,%rax), %ecx

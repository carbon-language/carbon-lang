; RUN: llc < %s -march=x86 -mtriple=i386-linux-gnu > %t
; RUN: grep {movswl	%gs:i@NTPOFF, %eax} %t
; RUN: grep {movzwl	%gs:j@NTPOFF, %eax} %t
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-linux-gnu > %t2
; RUN: grep {movswl	%fs:i@TPOFF, %edi} %t2
; RUN: grep {movzwl	%fs:j@TPOFF, %edi} %t2

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

; RUN: llc < %s -mtriple=x86_64-pc-win32 | FileCheck %s

; Verify that the var arg parameters which are passed in registers are stored
; in home stack slots allocated by the caller and that AP is correctly
; calculated.
define void @average_va(i32 %count, ...) nounwind {
entry:
; CHECK: subq	$40, %rsp
; CHECK: movq	%r9, 72(%rsp)
; CHECK: movq	%r8, 64(%rsp)
; CHECK: movq	%rdx, 56(%rsp)
; CHECK: leaq	56(%rsp), %rax

  %ap = alloca i8*, align 8                       ; <i8**> [#uses=1]
  %ap1 = bitcast i8** %ap to i8*                  ; <i8*> [#uses=1]
  call void @llvm.va_start(i8* %ap1)
  ret void
}

declare void @llvm.va_start(i8*) nounwind

; RUN: llc < %s -o - -march=x86-64 | FileCheck %s
; PR8573

; CHECK: _foo:
; CHECK: leaq    (%rdi), %rax
; CHECK-NEXT: movl    %esi, %ecx
; CHECK-NEXT: monitor
define void @foo(i8* %P, i32 %E, i32 %H) nounwind {
entry:
  tail call void @llvm.x86.sse3.monitor(i8* %P, i32 %E, i32 %H)
  ret void
}

declare void @llvm.x86.sse3.monitor(i8*, i32, i32) nounwind

; CHECK: _bar:
; CHECK: movl    %edi, %ecx
; CHECK-NEXT: movl    %esi, %eax
; CHECK-NEXT: mwait
define void @bar(i32 %E, i32 %H) nounwind {
entry:
  tail call void @llvm.x86.sse3.mwait(i32 %E, i32 %H)
  ret void
}

declare void @llvm.x86.sse3.mwait(i32, i32) nounwind

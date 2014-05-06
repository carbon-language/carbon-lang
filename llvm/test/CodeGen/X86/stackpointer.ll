; RUN: llc < %s -mtriple=x86_64-apple-darwin  | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux-gnueabi | FileCheck %s

define i64 @get_stack() nounwind {
entry:
; CHECK-LABEL: get_stack:
; CHECK: movq	%rsp, %rax
	%sp = call i64 @llvm.read_register.i64(metadata !0)
  ret i64 %sp
}

define void @set_stack(i64 %val) nounwind {
entry:
; CHECK-LABEL: set_stack:
; CHECK: movq	%rdi, %rsp
  call void @llvm.write_register.i64(metadata !0, i64 %val)
  ret void
}

declare i64 @llvm.read_register.i64(metadata) nounwind
declare void @llvm.write_register.i64(metadata, i64) nounwind

; register unsigned long current_stack_pointer asm("rsp");
; CHECK-NOT: .asciz  "rsp"
!0 = metadata !{metadata !"rsp\00"}

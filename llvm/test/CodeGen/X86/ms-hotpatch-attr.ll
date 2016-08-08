; RUN: llc < %s -march=x86 -filetype=asm | FileCheck -check-prefix=CHECK-32 %s
; RUN: llc < %s -march=x86-64 -filetype=asm | FileCheck -check-prefix=CHECK-64 %s
; RUN: llc < %s -mtriple=i386-windows-msvc -filetype=asm | FileCheck -check-prefix=MSVC-32 %s
; RUN: llc < %s -mtriple=x86_64-windows-msvc -filetype=asm | FileCheck -check-prefix=MSVC-64 %s

; CHECK-32: .space 64,204
; CHECK-32: .p2align 4, 0x90
; CHECK-32-LABEL: foo:
; CHECK-32: movl %edi, %edi
; CHECK-32-NEXT: pushl %ebp
; CHECK-32-NEXT: movl %esp, %ebp
; CHECK-64: .space 128,204
; CHECK-64: .p2align 4, 0x90
; CHECK-64-LABEL: foo:
; CHECK-64: xchgw %ax, %ax
; MSVC-32-NOT: .space 64,204
; MSVC-32-LABEL: _foo:
; MSVC-32: movl %edi, %edi
; MSVC-32-NEXT: pushl %ebp
; MSVC-32-NEXT: movl %esp, %ebp
; MSVC-64-NOT: .space 128,204
; MSVC-64-LABEL: foo:
; MSVC-64: xchgw %ax, %ax
define void @foo() nounwind "patchable-function"="ms-hotpatch" {
entry:
  ret void
}

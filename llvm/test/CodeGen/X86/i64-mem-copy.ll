; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=sse2 | FileCheck %s --check-prefix=X64
; RUN: llc < %s -mtriple=i386-unknown-unknown -mattr=sse2 | FileCheck %s --check-prefix=X32

; Use movq or movsd to load / store i64 values if sse2 is available.
; rdar://6659858

define void @foo(i64* %x, i64* %y) {
; X64-LABEL: foo:
; X64:       # BB#0:
; X64-NEXT:    movq (%rsi), %rax
; X64-NEXT:    movq %rax, (%rdi)
; X64-NEXT:    retq
;
; X32-LABEL: foo:
; X32:       # BB#0:
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X32-NEXT:    movsd {{.*#+}} xmm0 = mem[0],zero
; X32-NEXT:    movsd %xmm0, (%eax)
; X32-NEXT:    retl

  %tmp1 = load i64, i64* %y, align 8
  store i64 %tmp1, i64* %x, align 8
  ret void
}


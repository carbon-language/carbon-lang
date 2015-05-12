; RUN: llc -mtriple=x86_64-apple-darwin8 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-pc-linux < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin8 -terminal-rule < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-pc-linux -terminal-rule < %s | FileCheck %s

; CHECK-LABEL: return32
; CHECK-DAG: movq	$0, (%rdi)
; CHECK-DAG: movq	%rdi, %rax
; CHECK: retq
define i256 @return32() {
  ret i256 0
}

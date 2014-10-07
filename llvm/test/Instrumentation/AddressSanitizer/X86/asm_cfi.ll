; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=+sse2 -asm-instrumentation=address -asan-instrument-assembly | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: mov8b_rbp
; CHECK: pushq %rbp
; CHECK-NOT: .cfi_adjust_cfa_offset 8
; CHECK: movq %rbp, %rbp
; CHECK: .cfi_remember_state
; CHECK: .cfi_def_cfa_register %rbp
; CHECK: leaq -128(%rsp)
; CHECK: callq __asan_report_load8@PLT
; CHECK: leaq 128(%rsp)
; CHECK: popq %rbp
; CHECK: .cfi_restore_state
; CHECK-NOT: .cfi_adjust_cfa_offset -8
; CHECK: retq
define void @mov8b_rbp(i64* %dst, i64* %src) #0 {
entry:
  tail call void asm sideeffect "movq ($0), %rax \0A\09movq %rax, ($1) \0A\09", "r,r,~{rax},~{memory},~{dirflag},~{fpsr},~{flags}"(i64* %src, i64* %dst)
  ret void
}

; CHECK-LABEL: mov8b_rsp
; CHECK: pushq %rbp
; CHECK: .cfi_adjust_cfa_offset 8
; CHECK: movq %rsp, %rbp
; CHECK: .cfi_remember_state
; CHECK: .cfi_def_cfa_register %rbp
; CHECK: leaq -128(%rsp)
; CHECK: callq __asan_report_load8@PLT
; CHECK: leaq 128(%rsp)
; CHECK: popq %rbp
; CHECK: .cfi_restore_state
; CHECK: .cfi_adjust_cfa_offset -8
; CHECK: retq
define void @mov8b_rsp(i64* %dst, i64* %src) #1 {
entry:
  tail call void asm sideeffect "movq ($0), %rax \0A\09movq %rax, ($1) \0A\09", "r,r,~{rax},~{memory},~{dirflag},~{fpsr},~{flags}"(i64* %src, i64* %dst)
  ret void
}

; CHECK-LABEL: mov8b_rsp_no_cfi
; CHECK-NOT: .cfi{{[a-z_]+}}
define void @mov8b_rsp_no_cfi(i64* %dst, i64* %src) #2 {
entry:
  tail call void asm sideeffect "movq ($0), %rax \0A\09movq %rax, ($1) \0A\09", "r,r,~{rax},~{memory},~{dirflag},~{fpsr},~{flags}"(i64* %src, i64* %dst)
  ret void
}

attributes #0 = { nounwind sanitize_address uwtable "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind sanitize_address uwtable "no-frame-pointer-elim"="false" }
attributes #2 = { nounwind sanitize_address "no-frame-pointer-elim"="false" }

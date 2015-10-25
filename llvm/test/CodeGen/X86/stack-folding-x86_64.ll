; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

;TODO stack_fold_bsf_i16
declare i16 @llvm.cttz.i16(i16, i1)

define i32 @stack_fold_bsf_i32(i32 %a0) {
  ;CHECK-LABEL: stack_fold_bsf_i32
  ;CHECK:       bsfl {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = call i32 @llvm.cttz.i32(i32 %a0, i1 -1)
  ret i32 %2
}
declare i32 @llvm.cttz.i32(i32, i1)

define i64 @stack_fold_bsf_i64(i64 %a0) {
  ;CHECK-LABEL: stack_fold_bsf_i64
  ;CHECK:       bsfq {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = call i64 @llvm.cttz.i64(i64 %a0, i1 -1)
  ret i64 %2
}
declare i64 @llvm.cttz.i64(i64, i1)

;TODO stack_fold_bsr_i16
declare i16 @llvm.ctlz.i16(i16, i1)

define i32 @stack_fold_bsr_i32(i32 %a0) {
  ;CHECK-LABEL: stack_fold_bsr_i32
  ;CHECK:       bsrl {{-?[0-9]*}}(%rsp), %eax {{.*#+}} 4-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = call i32 @llvm.ctlz.i32(i32 %a0, i1 -1)
  ret i32 %2
}
declare i32 @llvm.ctlz.i32(i32, i1)

define i64 @stack_fold_bsr_i64(i64 %a0) {
  ;CHECK-LABEL: stack_fold_bsr_i64
  ;CHECK:       bsrq {{-?[0-9]*}}(%rsp), %rax {{.*#+}} 8-byte Folded Reload
  %1 = tail call i64 asm sideeffect "nop", "=x,~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %2 = call i64 @llvm.ctlz.i64(i64 %a0, i1 -1)
  ret i64 %2
}
declare i64 @llvm.ctlz.i64(i64, i1)

# RUN: llvm-mc -filetype=obj -triple x86_64 --x86-align-branch-boundary=32 --x86-align-branch=call+jmp+indirect+ret+jcc+fused %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

  # This file includes cases which are problematic to apply automatic padding
  # in the assembler.  These are the examples brought up in review and
  # discussion which motivate the need for a assembler directive to
  # selectively enable/disable.
  # FIXME: the checks are checking current *incorrect* behavior

  .text

  # In the first test, we have a label which is expected to be bound to the
  # start of the call.  For instance, we want to associate a fault on the call
  # target with some bit of higher level sementic.
  # CHECK-LABEL: <labeled_call_test1>:
  # CHECK: 1f <label_before>:
  # CHECK: 1f: nop
  # CHECK: 20: callq
  .globl  labeled_call_test1
  .p2align  5
labeled_call_test1:
  .rept 31
  int3
  .endr
label_before:
  callq bar

  # Our second test is like the first w/a labeled fault, but specifically to
  # a fused memory comparison.  This is the form produced by implicit null
  # checks for instance.
  # CHECK-LABEL: <implicit_null_check>:
  # CHECK: 5f <fault_addr>:
  # CHECK: 5f: nop
  # CHECK: 60: cmpq
  .globl  implicit_null_check
  .p2align  5
implicit_null_check:
  .rept 31
  int3
  .endr
fault_addr:
  cmpq (%rsi), %rdi
  jne bar

  .p2align 4
  .type   bar,@function
bar:
  retq

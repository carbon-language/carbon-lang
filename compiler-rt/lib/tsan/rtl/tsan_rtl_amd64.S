.section .text

.hidden __tsan_trace_switch
.globl __tsan_trace_switch_thunk
__tsan_trace_switch_thunk:
  .cfi_startproc
  # Save scratch registers.
  push %rax
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rax, 0
  push %rcx
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rcx, 0
  push %rdx
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rdx, 0
  push %rsi
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rsi, 0
  push %rdi
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rdi, 0
  push %r8
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %r8, 0
  push %r9
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %r9, 0
  push %r10
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %r10, 0
  push %r11
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %r11, 0
  # Align stack frame.
  push %rbx  # non-scratch
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rbx, 0
  mov %rsp, %rbx  # save current rsp
  .cfi_def_cfa_register %rbx
  shr $4, %rsp  # clear 4 lsb, align to 16
  shl $4, %rsp

  call __tsan_trace_switch

  # Unalign stack frame back.
  mov %rbx, %rsp  # restore the original rsp
  .cfi_def_cfa_register %rsp
  pop %rbx
  .cfi_adjust_cfa_offset -8
  # Restore scratch registers.
  pop %r11
  .cfi_adjust_cfa_offset -8
  pop %r10
  .cfi_adjust_cfa_offset -8
  pop %r9
  .cfi_adjust_cfa_offset -8
  pop %r8
  .cfi_adjust_cfa_offset -8
  pop %rdi
  .cfi_adjust_cfa_offset -8
  pop %rsi
  .cfi_adjust_cfa_offset -8
  pop %rdx
  .cfi_adjust_cfa_offset -8
  pop %rcx
  .cfi_adjust_cfa_offset -8
  pop %rax
  .cfi_adjust_cfa_offset -8
  .cfi_restore %rax
  .cfi_restore %rbx
  .cfi_restore %rcx
  .cfi_restore %rdx
  .cfi_restore %rsi
  .cfi_restore %rdi
  .cfi_restore %r8
  .cfi_restore %r9
  .cfi_restore %r10
  .cfi_restore %r11
  ret
  .cfi_endproc

.hidden __tsan_report_race
.globl __tsan_report_race_thunk
__tsan_report_race_thunk:
  .cfi_startproc
  # Save scratch registers.
  push %rax
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rax, 0
  push %rcx
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rcx, 0
  push %rdx
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rdx, 0
  push %rsi
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rsi, 0
  push %rdi
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rdi, 0
  push %r8
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %r8, 0
  push %r9
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %r9, 0
  push %r10
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %r10, 0
  push %r11
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %r11, 0
  # Align stack frame.
  push %rbx  # non-scratch
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rbx, 0
  mov %rsp, %rbx  # save current rsp
  .cfi_def_cfa_register %rbx
  shr $4, %rsp  # clear 4 lsb, align to 16
  shl $4, %rsp

  call __tsan_report_race

  # Unalign stack frame back.
  mov %rbx, %rsp  # restore the original rsp
  .cfi_def_cfa_register %rsp
  pop %rbx
  .cfi_adjust_cfa_offset -8
  # Restore scratch registers.
  pop %r11
  .cfi_adjust_cfa_offset -8
  pop %r10
  .cfi_adjust_cfa_offset -8
  pop %r9
  .cfi_adjust_cfa_offset -8
  pop %r8
  .cfi_adjust_cfa_offset -8
  pop %rdi
  .cfi_adjust_cfa_offset -8
  pop %rsi
  .cfi_adjust_cfa_offset -8
  pop %rdx
  .cfi_adjust_cfa_offset -8
  pop %rcx
  .cfi_adjust_cfa_offset -8
  pop %rax
  .cfi_adjust_cfa_offset -8
  .cfi_restore %rax
  .cfi_restore %rbx
  .cfi_restore %rcx
  .cfi_restore %rdx
  .cfi_restore %rsi
  .cfi_restore %rdi
  .cfi_restore %r8
  .cfi_restore %r9
  .cfi_restore %r10
  .cfi_restore %r11
  ret
  .cfi_endproc

#ifdef __linux__
/* We do not need executable stack.  */
.section        .note.GNU-stack,"",@progbits
#endif

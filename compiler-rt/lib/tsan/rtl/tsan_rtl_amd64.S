.section .text

.globl __tsan_trace_switch_thunk
__tsan_trace_switch_thunk:
  # Save scratch registers.
  push %rax
  push %rcx
  push %rdx
  push %rsi
  push %rdi
  push %r8
  push %r9
  push %r10
  push %r11
  # Align stack frame.
  push %rbx  # non-scratch
  mov %rsp, %rbx  # save current rsp
  shr $4, %rsp  # clear 4 lsb, align to 16
  shl $4, %rsp

  call __tsan_trace_switch

  # Unalign stack frame back.
  mov %rbx, %rsp  # restore the original rsp
  pop %rbx
  # Restore scratch registers.
  pop %r11
  pop %r10
  pop %r9
  pop %r8
  pop %rdi
  pop %rsi
  pop %rdx
  pop %rcx
  pop %rax
  ret

.globl __tsan_report_race_thunk
__tsan_report_race_thunk:
  # Save scratch registers.
  push %rax
  push %rcx
  push %rdx
  push %rsi
  push %rdi
  push %r8
  push %r9
  push %r10
  push %r11
  # Align stack frame.
  push %rbx  # non-scratch
  mov %rsp, %rbx  # save current rsp
  shr $4, %rsp  # clear 4 lsb, align to 16
  shl $4, %rsp

  call __tsan_report_race

  # Unalign stack frame back.
  mov %rbx, %rsp  # restore the original rsp
  pop %rbx
  # Restore scratch registers.
  pop %r11
  pop %r10
  pop %r9
  pop %r8
  pop %rdi
  pop %rsi
  pop %rdx
  pop %rcx
  pop %rax
  ret

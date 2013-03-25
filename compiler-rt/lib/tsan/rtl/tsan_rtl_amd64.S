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

.hidden __tsan_setjmp
.comm _ZN14__interception11real_setjmpE,8,8
.globl setjmp
.type setjmp, @function
setjmp:
  .cfi_startproc
  // save env parameter
  push %rdi
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rdi, 0
  // obtain %rsp
  lea 16(%rsp), %rdi
  mov %rdi, %rsi
  xor %fs:0x30, %rsi  // magic mangling of rsp (see libc setjmp)
  rol $0x11, %rsi
  // call tsan interceptor
  call __tsan_setjmp
  // restore env parameter
  pop %rdi
  .cfi_adjust_cfa_offset -8
  .cfi_restore %rdi
  // tail jump to libc setjmp
  movl $0, %eax
  movq _ZN14__interception11real_setjmpE@GOTPCREL(%rip), %rdx
  jmp *(%rdx)
  .cfi_endproc
.size setjmp, .-setjmp

.comm _ZN14__interception12real__setjmpE,8,8
.globl _setjmp
.type _setjmp, @function
_setjmp:
  .cfi_startproc
  // save env parameter
  push %rdi
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rdi, 0
  // obtain %rsp
  lea 16(%rsp), %rdi
  mov %rdi, %rsi
  xor %fs:0x30, %rsi  // magic mangling of rsp (see libc setjmp)
  rol $0x11, %rsi
  // call tsan interceptor
  call __tsan_setjmp
  // restore env parameter
  pop %rdi
  .cfi_adjust_cfa_offset -8
  .cfi_restore %rdi
  // tail jump to libc setjmp
  movl $0, %eax
  movq _ZN14__interception12real__setjmpE@GOTPCREL(%rip), %rdx
  jmp *(%rdx)
  .cfi_endproc
.size _setjmp, .-_setjmp

.comm _ZN14__interception14real_sigsetjmpE,8,8
.globl sigsetjmp
.type sigsetjmp, @function
sigsetjmp:
  .cfi_startproc
  // save env parameter
  push %rdi
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rdi, 0
  // save savesigs parameter
  push %rsi
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rsi, 0
  // align stack frame
  sub $8, %rsp
  .cfi_adjust_cfa_offset 8
  // obtain %rsp
  lea 32(%rsp), %rdi
  mov %rdi, %rsi
  xor %fs:0x30, %rsi  // magic mangling of rsp (see libc setjmp)
  rol $0x11, %rsi
  // call tsan interceptor
  call __tsan_setjmp
  // unalign stack frame
  add $8, %rsp
  .cfi_adjust_cfa_offset -8
  // restore savesigs parameter
  pop %rsi
  .cfi_adjust_cfa_offset -8
  .cfi_restore %rsi
  // restore env parameter
  pop %rdi
  .cfi_adjust_cfa_offset -8
  .cfi_restore %rdi
  // tail jump to libc sigsetjmp
  movl $0, %eax
  movq _ZN14__interception14real_sigsetjmpE@GOTPCREL(%rip), %rdx
  jmp *(%rdx)
  .cfi_endproc
.size sigsetjmp, .-sigsetjmp

.comm _ZN14__interception16real___sigsetjmpE,8,8
.globl __sigsetjmp
.type __sigsetjmp, @function
__sigsetjmp:
  .cfi_startproc
  // save env parameter
  push %rdi
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rdi, 0
  // save savesigs parameter
  push %rsi
  .cfi_adjust_cfa_offset 8
  .cfi_rel_offset %rsi, 0
  // align stack frame
  sub $8, %rsp
  .cfi_adjust_cfa_offset 8
  // obtain %rsp
  lea 32(%rsp), %rdi
  mov %rdi, %rsi
  xor %fs:0x30, %rsi  // magic mangling of rsp (see libc setjmp)
  rol $0x11, %rsi
  // call tsan interceptor
  call __tsan_setjmp
  // unalign stack frame
  add $8, %rsp
  .cfi_adjust_cfa_offset -8
  // restore savesigs parameter
  pop %rsi
  .cfi_adjust_cfa_offset -8
  .cfi_restore %rsi
  // restore env parameter
  pop %rdi
  .cfi_adjust_cfa_offset -8
  .cfi_restore %rdi
  // tail jump to libc sigsetjmp
  movl $0, %eax
  movq _ZN14__interception16real___sigsetjmpE@GOTPCREL(%rip), %rdx
  jmp *(%rdx)
  .cfi_endproc
.size __sigsetjmp, .-__sigsetjmp

#ifdef __linux__
/* We do not need executable stack.  */
.section        .note.GNU-stack,"",@progbits
#endif

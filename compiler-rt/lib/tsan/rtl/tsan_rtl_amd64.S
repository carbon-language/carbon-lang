#include "sanitizer_common/sanitizer_asm.h"
.section .text

.hidden __tsan_trace_switch
.globl __tsan_trace_switch_thunk
__tsan_trace_switch_thunk:
  CFI_STARTPROC
  # Save scratch registers.
  push %rax
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rax, 0)
  push %rcx
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rcx, 0)
  push %rdx
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rdx, 0)
  push %rsi
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rsi, 0)
  push %rdi
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rdi, 0)
  push %r8
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%r8, 0)
  push %r9
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%r9, 0)
  push %r10
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%r10, 0)
  push %r11
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%r11, 0)
  # Align stack frame.
  push %rbx  # non-scratch
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rbx, 0)
  mov %rsp, %rbx  # save current rsp
  CFI_DEF_CFA_REGISTER(%rbx)
  shr $4, %rsp  # clear 4 lsb, align to 16
  shl $4, %rsp

  call __tsan_trace_switch

  # Unalign stack frame back.
  mov %rbx, %rsp  # restore the original rsp
  CFI_DEF_CFA_REGISTER(%rsp)
  pop %rbx
  CFI_ADJUST_CFA_OFFSET(-8)
  # Restore scratch registers.
  pop %r11
  CFI_ADJUST_CFA_OFFSET(-8)
  pop %r10
  CFI_ADJUST_CFA_OFFSET(-8)
  pop %r9
  CFI_ADJUST_CFA_OFFSET(-8)
  pop %r8
  CFI_ADJUST_CFA_OFFSET(-8)
  pop %rdi
  CFI_ADJUST_CFA_OFFSET(-8)
  pop %rsi
  CFI_ADJUST_CFA_OFFSET(-8)
  pop %rdx
  CFI_ADJUST_CFA_OFFSET(-8)
  pop %rcx
  CFI_ADJUST_CFA_OFFSET(-8)
  pop %rax
  CFI_ADJUST_CFA_OFFSET(-8)
  CFI_RESTORE(%rax)
  CFI_RESTORE(%rbx)
  CFI_RESTORE(%rcx)
  CFI_RESTORE(%rdx)
  CFI_RESTORE(%rsi)
  CFI_RESTORE(%rdi)
  CFI_RESTORE(%r8)
  CFI_RESTORE(%r9)
  CFI_RESTORE(%r10)
  CFI_RESTORE(%r11)
  ret
  CFI_ENDPROC

.hidden __tsan_report_race
.globl __tsan_report_race_thunk
__tsan_report_race_thunk:
  CFI_STARTPROC
  # Save scratch registers.
  push %rax
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rax, 0)
  push %rcx
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rcx, 0)
  push %rdx
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rdx, 0)
  push %rsi
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rsi, 0)
  push %rdi
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rdi, 0)
  push %r8
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%r8, 0)
  push %r9
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%r9, 0)
  push %r10
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%r10, 0)
  push %r11
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%r11, 0)
  # Align stack frame.
  push %rbx  # non-scratch
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rbx, 0)
  mov %rsp, %rbx  # save current rsp
  CFI_DEF_CFA_REGISTER(%rbx)
  shr $4, %rsp  # clear 4 lsb, align to 16
  shl $4, %rsp

  call __tsan_report_race

  # Unalign stack frame back.
  mov %rbx, %rsp  # restore the original rsp
  CFI_DEF_CFA_REGISTER(%rsp)
  pop %rbx
  CFI_ADJUST_CFA_OFFSET(-8)
  # Restore scratch registers.
  pop %r11
  CFI_ADJUST_CFA_OFFSET(-8)
  pop %r10
  CFI_ADJUST_CFA_OFFSET(-8)
  pop %r9
  CFI_ADJUST_CFA_OFFSET(-8)
  pop %r8
  CFI_ADJUST_CFA_OFFSET(-8)
  pop %rdi
  CFI_ADJUST_CFA_OFFSET(-8)
  pop %rsi
  CFI_ADJUST_CFA_OFFSET(-8)
  pop %rdx
  CFI_ADJUST_CFA_OFFSET(-8)
  pop %rcx
  CFI_ADJUST_CFA_OFFSET(-8)
  pop %rax
  CFI_ADJUST_CFA_OFFSET(-8)
  CFI_RESTORE(%rax)
  CFI_RESTORE(%rbx)
  CFI_RESTORE(%rcx)
  CFI_RESTORE(%rdx)
  CFI_RESTORE(%rsi)
  CFI_RESTORE(%rdi)
  CFI_RESTORE(%r8)
  CFI_RESTORE(%r9)
  CFI_RESTORE(%r10)
  CFI_RESTORE(%r11)
  ret
  CFI_ENDPROC

.hidden __tsan_setjmp
.comm _ZN14__interception11real_setjmpE,8,8
.globl setjmp
.type setjmp, @function
setjmp:
  CFI_STARTPROC
  // save env parameter
  push %rdi
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rdi, 0)
  // obtain %rsp
#if defined(__FreeBSD__)
  lea 8(%rsp), %rdi
  mov %rdi, %rsi
#else
  lea 16(%rsp), %rdi
  mov %rdi, %rsi
  xor %fs:0x30, %rsi  // magic mangling of rsp (see libc setjmp)
  rol $0x11, %rsi
#endif
  // call tsan interceptor
  call __tsan_setjmp
  // restore env parameter
  pop %rdi
  CFI_ADJUST_CFA_OFFSET(-8)
  CFI_RESTORE(%rdi)
  // tail jump to libc setjmp
  movl $0, %eax
  movq _ZN14__interception11real_setjmpE@GOTPCREL(%rip), %rdx
  jmp *(%rdx)
  CFI_ENDPROC
.size setjmp, .-setjmp

.comm _ZN14__interception12real__setjmpE,8,8
.globl _setjmp
.type _setjmp, @function
_setjmp:
  CFI_STARTPROC
  // save env parameter
  push %rdi
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rdi, 0)
  // obtain %rsp
#if defined(__FreeBSD__)
  lea 8(%rsp), %rdi
  mov %rdi, %rsi
#else
  lea 16(%rsp), %rdi
  mov %rdi, %rsi
  xor %fs:0x30, %rsi  // magic mangling of rsp (see libc setjmp)
  rol $0x11, %rsi
#endif
  // call tsan interceptor
  call __tsan_setjmp
  // restore env parameter
  pop %rdi
  CFI_ADJUST_CFA_OFFSET(-8)
  CFI_RESTORE(%rdi)
  // tail jump to libc setjmp
  movl $0, %eax
  movq _ZN14__interception12real__setjmpE@GOTPCREL(%rip), %rdx
  jmp *(%rdx)
  CFI_ENDPROC
.size _setjmp, .-_setjmp

.comm _ZN14__interception14real_sigsetjmpE,8,8
.globl sigsetjmp
.type sigsetjmp, @function
sigsetjmp:
  CFI_STARTPROC
  // save env parameter
  push %rdi
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rdi, 0)
  // save savesigs parameter
  push %rsi
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rsi, 0)
  // align stack frame
  sub $8, %rsp
  CFI_ADJUST_CFA_OFFSET(8)
  // obtain %rsp
#if defined(__FreeBSD__)
  lea 24(%rsp), %rdi
  mov %rdi, %rsi
#else
  lea 32(%rsp), %rdi
  mov %rdi, %rsi
  xor %fs:0x30, %rsi  // magic mangling of rsp (see libc setjmp)
  rol $0x11, %rsi
#endif
  // call tsan interceptor
  call __tsan_setjmp
  // unalign stack frame
  add $8, %rsp
  CFI_ADJUST_CFA_OFFSET(-8)
  // restore savesigs parameter
  pop %rsi
  CFI_ADJUST_CFA_OFFSET(-8)
  CFI_RESTORE(%rsi)
  // restore env parameter
  pop %rdi
  CFI_ADJUST_CFA_OFFSET(-8)
  CFI_RESTORE(%rdi)
  // tail jump to libc sigsetjmp
  movl $0, %eax
  movq _ZN14__interception14real_sigsetjmpE@GOTPCREL(%rip), %rdx
  jmp *(%rdx)
  CFI_ENDPROC
.size sigsetjmp, .-sigsetjmp

.comm _ZN14__interception16real___sigsetjmpE,8,8
.globl __sigsetjmp
.type __sigsetjmp, @function
__sigsetjmp:
  CFI_STARTPROC
  // save env parameter
  push %rdi
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rdi, 0)
  // save savesigs parameter
  push %rsi
  CFI_ADJUST_CFA_OFFSET(8)
  CFI_REL_OFFSET(%rsi, 0)
  // align stack frame
  sub $8, %rsp
  CFI_ADJUST_CFA_OFFSET(8)
  // obtain %rsp
#if defined(__FreeBSD__)
  lea 24(%rsp), %rdi
  mov %rdi, %rsi
#else
  lea 32(%rsp), %rdi
  mov %rdi, %rsi
  xor %fs:0x30, %rsi  // magic mangling of rsp (see libc setjmp)
  rol $0x11, %rsi
#endif
  // call tsan interceptor
  call __tsan_setjmp
  // unalign stack frame
  add $8, %rsp
  CFI_ADJUST_CFA_OFFSET(-8)
  // restore savesigs parameter
  pop %rsi
  CFI_ADJUST_CFA_OFFSET(-8)
  CFI_RESTORE(%rsi)
  // restore env parameter
  pop %rdi
  CFI_ADJUST_CFA_OFFSET(-8)
  CFI_RESTORE(%rdi)
  // tail jump to libc sigsetjmp
  movl $0, %eax
  movq _ZN14__interception16real___sigsetjmpE@GOTPCREL(%rip), %rdx
  jmp *(%rdx)
  CFI_ENDPROC
.size __sigsetjmp, .-__sigsetjmp

#if defined(__FreeBSD__) || defined(__linux__)
/* We do not need executable stack.  */
.section        .note.GNU-stack,"",@progbits
#endif

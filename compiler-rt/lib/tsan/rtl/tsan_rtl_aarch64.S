#include "sanitizer_common/sanitizer_asm.h"
.section .text

.hidden __tsan_setjmp
.comm _ZN14__interception11real_setjmpE,8,8
.type setjmp, @function
setjmp:
  CFI_STARTPROC

  // save env parameters for function call
  stp     x29, x30, [sp, -32]!
  CFI_DEF_CFA_OFFSET (32)
  CFI_OFFSET (29, -32)
  CFI_OFFSET (30, -24)

  // Adjust the SP for previous frame
  add     x29, sp, 0
  CFI_DEF_CFA_REGISTER (29)

  // Save jmp_buf
  str     x19, [sp, 16]
  CFI_OFFSET (19, -16)
  mov     x19, x0

  // SP pointer mangling (see glibc setjmp)
  adrp    x2, :got:__pointer_chk_guard
  ldr     x2, [x2, #:got_lo12:__pointer_chk_guard]
  add     x0, x29, 32
  ldr     x2, [x2]
  eor     x1, x2, x0

  // call tsan interceptor
  bl      __tsan_setjmp

  // restore env parameter
  mov     x0, x19
  ldr     x19, [sp, 16]
  ldp     x29, x30, [sp], 32
  CFI_RESTORE (30)
  CFI_RESTORE (19)
  CFI_DEF_CFA (31, 0)

  // tail jump to libc setjmp
  adrp    x1, :got:_ZN14__interception11real_setjmpE
  ldr     x1, [x1, #:got_lo12:_ZN14__interception11real_setjmpE]
  ldr     x1, [x1]
  br      x1

  CFI_ENDPROC
.size setjmp, .-setjmp

.comm _ZN14__interception12real__setjmpE,8,8
.globl _setjmp
.type _setjmp, @function
_setjmp:
  CFI_STARTPROC

  // save env parameters for function call
  stp     x29, x30, [sp, -32]!
  CFI_DEF_CFA_OFFSET (32)
  CFI_OFFSET (29, -32)
  CFI_OFFSET (30, -24)

  // Adjust the SP for previous frame
  add     x29, sp, 0
  CFI_DEF_CFA_REGISTER (29)

  // Save jmp_buf
  str     x19, [sp, 16]
  CFI_OFFSET (19, -16)
  mov     x19, x0

  // SP pointer mangling (see glibc setjmp)
  adrp    x2, :got:__pointer_chk_guard
  ldr     x2, [x2, #:got_lo12:__pointer_chk_guard]
  add     x0, x29, 32
  ldr     x2, [x2]
  eor     x1, x2, x0

  // call tsan interceptor
  bl      __tsan_setjmp

  // Restore jmp_buf parameter
  mov     x0, x19
  ldr     x19, [sp, 16]
  ldp     x29, x30, [sp], 32
  CFI_RESTORE (30)
  CFI_RESTORE (19)
  CFI_DEF_CFA (31, 0)

  // tail jump to libc setjmp
  adrp    x1, :got:_ZN14__interception12real__setjmpE
  ldr     x1, [x1, #:got_lo12:_ZN14__interception12real__setjmpE]
  ldr     x1, [x1]
  br      x1

  CFI_ENDPROC
.size _setjmp, .-_setjmp

.comm _ZN14__interception14real_sigsetjmpE,8,8
.globl sigsetjmp
.type sigsetjmp, @function
sigsetjmp:
  CFI_STARTPROC

  // save env parameters for function call
  stp     x29, x30, [sp, -32]!
  CFI_DEF_CFA_OFFSET (32)
  CFI_OFFSET (29, -32)
  CFI_OFFSET (30, -24)

  // Adjust the SP for previous frame
  add     x29, sp, 0
  CFI_DEF_CFA_REGISTER (29)

  // Save jmp_buf and savesigs
  stp     x19, x20, [sp, 16]
  CFI_OFFSET (19, -16)
  CFI_OFFSET (20, -8)
  mov     w20, w1
  mov     x19, x0

  // SP pointer mangling (see glibc setjmp)
  adrp    x2, :got:__pointer_chk_guard
  ldr     x2, [x2, #:got_lo12:__pointer_chk_guard]
  add     x0, x29, 32
  ldr     x2, [x2]
  eor     x1, x2, x0

  // call tsan interceptor
  bl      __tsan_setjmp

  // restore env parameter
  mov     w1, w20
  mov     x0, x19
  ldp     x19, x20, [sp, 16]
  ldp     x29, x30, [sp], 32
  CFI_RESTORE (30)
  CFI_RESTORE (29)
  CFI_RESTORE (19)
  CFI_RESTORE (20)
  CFI_DEF_CFA (31, 0)

  // tail jump to libc sigsetjmp
  adrp    x2, :got:_ZN14__interception14real_sigsetjmpE
  ldr     x2, [x2, #:got_lo12:_ZN14__interception14real_sigsetjmpE]
  ldr     x2, [x2]
  br      x2
  CFI_ENDPROC
.size sigsetjmp, .-sigsetjmp

.comm _ZN14__interception16real___sigsetjmpE,8,8
.globl __sigsetjmp
.type __sigsetjmp, @function
__sigsetjmp:
  CFI_STARTPROC

  // save env parameters for function call
  stp     x29, x30, [sp, -32]!
  CFI_DEF_CFA_OFFSET (32)
  CFI_OFFSET (29, -32)
  CFI_OFFSET (30, -24)

  // Adjust the SP for previous frame
  add     x29, sp, 0
  CFI_DEF_CFA_REGISTER (29)

  // Save jmp_buf and savesigs
  stp     x19, x20, [sp, 16]
  CFI_OFFSET (19, -16)
  CFI_OFFSET (20, -8)
  mov     w20, w1
  mov     x19, x0

  // SP pointer mangling (see glibc setjmp)
  adrp    x2, :got:__pointer_chk_guard
  ldr     x2, [x2, #:got_lo12:__pointer_chk_guard]
  add     x0, x29, 32
  ldr     x2, [x2]
  eor     x1, x2, x0

  // call tsan interceptor
  bl      __tsan_setjmp

  mov     w1, w20
  mov     x0, x19
  ldp     x19, x20, [sp, 16]
  ldp     x29, x30, [sp], 32
  CFI_RESTORE (30)
  CFI_RESTORE (29)
  CFI_RESTORE (19)
  CFI_RESTORE (20)
  CFI_DEF_CFA (31, 0)

  // tail jump to libc __sigsetjmp
  adrp    x2, :got:_ZN14__interception16real___sigsetjmpE
  ldr     x2, [x2, #:got_lo12:_ZN14__interception16real___sigsetjmpE]
  ldr     x2, [x2]
  br      x2
  CFI_ENDPROC
.size __sigsetjmp, .-__sigsetjmp

#if defined(__linux__)
/* We do not need executable stack.  */
.section        .note.GNU-stack,"",@progbits
#endif

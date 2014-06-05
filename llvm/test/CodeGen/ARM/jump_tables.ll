; RUN: llc <%s -march=arm -jump-table-type=single | FileCheck --check-prefix=ARM %s
; RUN: llc <%s -march=thumb -jump-table-type=single | FileCheck --check-prefix=THUMB %s

define void @indirect_fun() unnamed_addr jumptable {
  ret void
}
define void ()* @get_fun() {
  ret void ()* @indirect_fun

; ARM:         ldr     r0, [[LABEL:.*]]
; ARM:         mov     pc, lr
; ARM: [[LABEL]]:
; ARM:         .long   __llvm_jump_instr_table_0_1

; THUMB:         ldr     r0, [[LABEL:.*]]
; THUMB:         bx      lr
; THUMB: [[LABEL]]:
; THUMB:         .long   __llvm_jump_instr_table_0_1
}

; ARM:         .globl  __llvm_jump_instr_table_0_1
; ARM:         .align  3
; ARM:         .type   __llvm_jump_instr_table_0_1,%function
; ARM: __llvm_jump_instr_table_0_1:
; ARM:         b     indirect_fun(PLT)

; THUMB:         .globl  __llvm_jump_instr_table_0_1
; THUMB:         .align  3
; THUMB:         .thumb_func
; THUMB:         .type   __llvm_jump_instr_table_0_1,%function
; THUMB: __llvm_jump_instr_table_0_1:
; THUMB:         b     indirect_fun(PLT)

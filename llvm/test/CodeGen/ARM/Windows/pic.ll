; RUN: llc -mtriple thumbv7-windows-itanium -relocation-model pic -filetype asm -o - %s \
; RUN:    | FileCheck %s

@external = external global i8

define arm_aapcs_vfpcc i8 @return_external() {
entry:
  %0 = load i8, i8* @external, align 1
  ret i8 %0
}

; CHECK-LABEL: return_external
; CHECK: movw r0, :lower16:external
; CHECK: movt r0, :upper16:external
; CHECK: ldrb r0, [r0]


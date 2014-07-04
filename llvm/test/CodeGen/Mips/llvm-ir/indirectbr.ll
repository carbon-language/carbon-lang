; Test all important variants of the unconditional 'br' instruction.

; RUN: llc -march=mips   -mcpu=mips32   < %s | FileCheck %s -check-prefix=ALL
; RUN: llc -march=mips   -mcpu=mips32r2 < %s | FileCheck %s -check-prefix=ALL
; RUN: llc -march=mips64 -mcpu=mips4    < %s | FileCheck %s -check-prefix=ALL
; RUN: llc -march=mips64 -mcpu=mips64   < %s | FileCheck %s -check-prefix=ALL
; RUN: llc -march=mips64 -mcpu=mips64r2 < %s | FileCheck %s -check-prefix=ALL

define i32 @br(i8 *%addr) {
; ALL-LABEL: br:
; ALL:           jr $4
; ALL: $BB0_1: # %L1
; ALL:           jr $ra
; ALL:           addiu $2, $zero, 0
; ALL: $BB0_2: # %L2
; ALL:           jr $ra
; ALL:           addiu $2, $zero, 1

entry:
  indirectbr i8* %addr, [label %L1, label %L2]

L1:
  ret i32 0

L2:
  ret i32 1
}

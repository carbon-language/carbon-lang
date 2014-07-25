; RUN: llc -march thumb -no-integrated-as %s -o - | FileCheck %s --check-prefix=THUMB
; RUN: llc -march arm -no-integrated-as %s -o - | FileCheck %s --check-prefix=ARM

; In thumb mode, emit ".code 16" before global inline-asm instructions.

; THUMB: .code 16
; THUMB: stmib
; THUMB: .code 16

; ARM-NOT: .code 16
; ARM:     stmib

module asm "stmib sp, {r0-r14};"

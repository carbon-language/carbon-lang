; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck -check-prefix=CHECK-INST %s


foo:

  ror r31
  ror r25
  ror r5
  ror r0

; CHECK: ror r31                ; encoding: [0xf7,0x95]
; CHECK: ror r25                ; encoding: [0x97,0x95]
; CHECK: ror r5                 ; encoding: [0x57,0x94]
; CHECK: ror r0                 ; encoding: [0x07,0x94]

; CHECK-INST: ror r31
; CHECK-INST: ror r25
; CHECK-INST: ror r5
; CHECK-INST: ror r0

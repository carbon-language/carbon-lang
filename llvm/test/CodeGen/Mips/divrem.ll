; RUN: llc -march=mips -mcpu=mips32 -verify-machineinstrs < %s | FileCheck %s -check-prefix=ALL -check-prefix=ACC -check-prefix=TRAP
; RUN: llc -march=mips -mcpu=mips32 -mno-check-zero-division < %s | FileCheck %s -check-prefix=ALL -check-prefix=ACC -check-prefix=NOCHECK

; FileCheck Prefixes:
;   ALL - All targets
;   ACC - Accumulator based multiply/divide. I.e. All ISA's before MIPS32r6
;   TRAP - Division must be explicitly checked for divide by zero
;   NOCHECK - Division by zero will not be detected

@g0 = common global i32 0, align 4
@g1 = common global i32 0, align 4

define i32 @sdiv1(i32 %a0, i32 %a1) nounwind readnone {
entry:
; ALL-LABEL: sdiv1:

; ACC:           div $zero, $4, $5

; TRAP:          teq $5, $zero, 7
; NOCHECK-NOT:   teq

; ACC:           mflo $2

; ALL: .end sdiv1

  %div = sdiv i32 %a0, %a1
  ret i32 %div
}

define i32 @srem1(i32 %a0, i32 %a1) nounwind readnone {
entry:
; ALL-LABEL: srem1:

; ACC:           div $zero, $4, $5

; TRAP:          teq $5, $zero, 7
; NOCHECK-NOT:   teq

; ACC:           mfhi $2

; ALL: .end srem1

  %rem = srem i32 %a0, %a1
  ret i32 %rem
}

define i32 @udiv1(i32 %a0, i32 %a1) nounwind readnone {
entry:
; ALL-LABEL: udiv1:

; ACC:           divu $zero, $4, $5

; TRAP:          teq $5, $zero, 7
; NOCHECK-NOT:   teq

; ACC:           mflo $2

; ALL: .end udiv1
  %div = udiv i32 %a0, %a1
  ret i32 %div
}

define i32 @urem1(i32 %a0, i32 %a1) nounwind readnone {
entry:
; ALL-LABEL: urem1:

; ACC:           divu $zero, $4, $5

; TRAP:          teq $5, $zero, 7
; NOCHECK-NOT:   teq

; ACC:           mfhi $2

; ALL: .end urem1

  %rem = urem i32 %a0, %a1
  ret i32 %rem
}

define i32 @sdivrem1(i32 %a0, i32 %a1, i32* nocapture %r) nounwind {
entry:
; ALL-LABEL: sdivrem1:

; ACC:           div $zero, $4, $5
; TRAP:          teq $5, $zero, 7
; NOCHECK-NOT:   teq
; ACC:           mflo $2
; ACC:           mfhi $[[R0:[0-9]+]]
; ACC:           sw $[[R0]], 0(${{[0-9]+}})

; ALL: .end sdivrem1

  %rem = srem i32 %a0, %a1
  store i32 %rem, i32* %r, align 4
  %div = sdiv i32 %a0, %a1
  ret i32 %div
}

define i32 @udivrem1(i32 %a0, i32 %a1, i32* nocapture %r) nounwind {
entry:
; ALL-LABEL: udivrem1:

; ACC:           divu $zero, $4, $5
; TRAP:          teq $5, $zero, 7
; NOCHECK-NOT:   teq
; ACC:           mflo $2
; ACC:           mfhi $[[R0:[0-9]+]]
; ACC:           sw $[[R0]], 0(${{[0-9]+}})

; ALL: .end udivrem1

  %rem = urem i32 %a0, %a1
  store i32 %rem, i32* %r, align 4
  %div = udiv i32 %a0, %a1
  ret i32 %div
}

; FIXME: It's not clear what this is supposed to test.
define i32 @killFlags() {
entry:
  %0 = load i32* @g0, align 4
  %1 = load i32* @g1, align 4
  %div = sdiv i32 %0, %1
  ret i32 %div
}

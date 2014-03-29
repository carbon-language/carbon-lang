; RUN: not llvm-mc -triple arm64-apple-darwin -show-encoding < %s 2> %t | FileCheck %s
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

foo:

; The first should encode as an expression. The second should error expecting
; a register.
  ldr x3, (foo + 4)
  ldr x3, [foo + 4]
; CHECK:  ldr x3, foo+4               ; encoding: [0bAAA00011,A,A,0x58]
; CHECK:                              ;   fixup A - offset: 0, value: foo+4, kind: fixup_arm64_pcrel_imm19
; CHECK-ERRORS: error: register expected

; The last argument should be flagged as an error.  rdar://9576009
  ld4.8b	{v0, v1, v2, v3}, [x0], #33
; CHECK-ERRORS: error: invalid operand for instruction
; CHECK-ERRORS: ld4.8b	{v0, v1, v2, v3}, [x0], #33


        ldr x0, [x0, #804]
        ldr w0, [x0, #802]
        ldr x0, [x0, #804]!
        ldr w0, [w0, #301]!
        ldr x0, [x0], #804
        ldr w0, [w0], #301

        ldp w3, w4, [x5, #11]!
        ldp x3, x4, [x5, #12]!
        ldp q3, q4, [x5, #12]!
        ldp w3, w4, [x5], #11
        ldp x3, x4, [x5], #12
        ldp q3, q4, [x5], #12

        ldur x0, [x1, #-257]

; CHECK-ERRORS: error: index must be a multiple of 8 in range [0,32760].
; CHECK-ERRORS:         ldr x0, [x0, #804]
; CHECK-ERRORS:                 ^
; CHECK-ERRORS: error: index must be a multiple of 4 in range [0,16380].
; CHECK-ERRORS:         ldr w0, [x0, #802]
; CHECK-ERRORS:                 ^
; CHECK-ERRORS: error: index must be an integer in range [-256,255].
; CHECK-ERRORS:         ldr x0, [x0, #804]!
; CHECK-ERRORS:                 ^
; CHECK-ERRORS: error: index must be an integer in range [-256,255].
; CHECK-ERRORS:         ldr w0, [w0, #301]!
; CHECK-ERRORS:                 ^
; CHECK-ERRORS: error: index must be an integer in range [-256,255].
; CHECK-ERRORS:         ldr x0, [x0], #804
; CHECK-ERRORS:                       ^
; CHECK-ERRORS: error: index must be an integer in range [-256,255].
; CHECK-ERRORS:         ldr w0, [w0], #301
; CHECK-ERRORS:                       ^
; CHECK-ERRORS: error: index must be a multiple of 4 in range [-256,252].
; CHECK-ERRORS:         ldp w3, w4, [x5, #11]!
; CHECK-ERRORS:                     ^
; CHECK-ERRORS: error: index must be a multiple of 8 in range [-512,504].
; CHECK-ERRORS:         ldp x3, x4, [x5, #12]!
; CHECK-ERRORS:                     ^
; CHECK-ERRORS: error: index must be a multiple of 16 in range [-1024,1008].
; CHECK-ERRORS:         ldp q3, q4, [x5, #12]!
; CHECK-ERRORS:                     ^
; CHECK-ERRORS: error: index must be a multiple of 4 in range [-256,252].
; CHECK-ERRORS:         ldp w3, w4, [x5], #11
; CHECK-ERRORS:                           ^
; CHECK-ERRORS: error: index must be a multiple of 8 in range [-512,504].
; CHECK-ERRORS:         ldp x3, x4, [x5], #12
; CHECK-ERRORS:                           ^
; CHECK-ERRORS: error: index must be a multiple of 8 in range [-512,504].
; CHECK-ERRORS:         ldp q3, q4, [x5], #12
; CHECK-ERRORS:                           ^
; CHECK-ERRORS: error: index must be an integer in range [-256,255].
; CHECK-ERRORS:         ldur x0, [x1, #-257]
; CHECK-ERRORS:                   ^



; Shift immediates range checking.
  sqrshrn b4, h9, #10
  rshrn v9.8b, v11.8h, #17
  sqrshrn v7.4h, v8.4s, #39
  uqshrn2 v4.4s, v5.2d, #67

; CHECK-ERRORS: error: immediate must be an integer in range [1,8].
; CHECK-ERRORS:   sqrshrn b4, h9, #10
; CHECK-ERRORS:                   ^
; CHECK-ERRORS: error: immediate must be an integer in range [1,8].
; CHECK-ERRORS:   rshrn v9.8b, v11.8h, #17
; CHECK-ERRORS:                        ^
; CHECK-ERRORS: error: immediate must be an integer in range [1,16].
; CHECK-ERRORS:   sqrshrn v7.4h, v8.4s, #39
; CHECK-ERRORS:                         ^
; CHECK-ERRORS: error: immediate must be an integer in range [1,32].
; CHECK-ERRORS:   uqshrn2 v4.4s, v5.2d, #67
; CHECK-ERRORS:                         ^


  st1.s4 {v14, v15}, [x2], #32
; CHECK-ERRORS: error: invalid type suffix for instruction
; CHECK-ERRORS: st1.s4 {v14, v15}, [x2], #32
; CHECK-ERRORS:     ^



; Load pair instructions where Rt==Rt2 and writeback load/store instructions
; where Rt==Rn or Rt2==Rn are unpredicatable.
  ldp x1, x2, [x2], #16
  ldp x2, x2, [x2], #16
  ldp w1, w2, [x2], #16
  ldp w2, w2, [x2], #16
  ldp x1, x1, [x2]

  ldr x2, [x2], #8
  ldr x2, [x2, #8]!
  ldr w2, [x2], #8
  ldr w2, [x2, #8]!

  str x2, [x2], #8
  str x2, [x2, #8]!
  str w2, [x2], #8
  str w2, [x2, #8]!

; CHECK-ERRORS: error: unpredictable LDP instruction, writeback base is also a destination
; CHECK-ERRORS:   ldp x1, x2, [x2], #16
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: unpredictable LDP instruction, writeback base is also a destination
; CHECK-ERRORS:   ldp x2, x2, [x2], #16
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable LDP instruction, writeback base is also a destination
; CHECK-ERRORS:   ldp w1, w2, [x2], #16
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: unpredictable LDP instruction, writeback base is also a destination
; CHECK-ERRORS:   ldp w2, w2, [x2], #16
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable LDP instruction, Rt2==Rt
; CHECK-ERRORS:   ldp x1, x1, [x2]
; CHECK-ERRORS:           ^
; CHECK-ERRORS: error: unpredictable LDR instruction, writeback base is also a source
; CHECK-ERRORS:   ldr x2, [x2], #8
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable LDR instruction, writeback base is also a source
; CHECK-ERRORS:   ldr x2, [x2, #8]!
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable LDR instruction, writeback base is also a source
; CHECK-ERRORS:   ldr w2, [x2], #8
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable LDR instruction, writeback base is also a source
; CHECK-ERRORS:   ldr w2, [x2, #8]!
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable STR instruction, writeback base is also a source
; CHECK-ERRORS:   str x2, [x2], #8
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable STR instruction, writeback base is also a source
; CHECK-ERRORS:   str x2, [x2, #8]!
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable STR instruction, writeback base is also a source
; CHECK-ERRORS:   str w2, [x2], #8
; CHECK-ERRORS:       ^
; CHECK-ERRORS: error: unpredictable STR instruction, writeback base is also a source
; CHECK-ERRORS:   str w2, [x2, #8]!
; CHECK-ERRORS:       ^

; The validity checking for shifted-immediate operands.  rdar://13174476
; Where the immediate is out of range.
  add w1, w2, w3, lsr #75

; CHECK-ERRORS: error: immediate value too large for shifter operand
; CHECK-ERRORS: add w1, w2, w3, lsr #75
; CHECK-ERRORS:                      ^

; logical instructions on 32-bit regs with shift > 31 is not legal
orr w0, w0, w0, lsl #32
; CHECK-ERRORS: error: shift value out of range
; CHECK-ERRORS:        orr w0, w0, w0, lsl #32
; CHECK-ERRORS:                        ^
eor w0, w0, w0, lsl #32
; CHECK-ERRORS: error: shift value out of range
; CHECK-ERRORS:        eor w0, w0, w0, lsl #32
; CHECK-ERRORS:                        ^
and w0, w0, w0, lsl #32
; CHECK-ERRORS: error: shift value out of range
; CHECK-ERRORS:        and w0, w0, w0, lsl #32
; CHECK-ERRORS:                        ^
ands w0, w0, w0, lsl #32
; CHECK-ERRORS: error: shift value out of range
; CHECK-ERRORS:        ands w0, w0, w0, lsl #32
; CHECK-ERRORS:                        ^

; Relocated expressions should not be accepted for 32-bit adds or sub (imm)
add w3, w5, sym@PAGEOFF
; CHECK-ERRORS: error: invalid immediate expression
; CHECK-ERRORS: add w3, w5, sym@PAGEOFF
; CHECK-ERRORS:             ^

adds w3, w5, sym@PAGEOFF
adds x9, x12, sym@PAGEOFF
; CHECK-ERRORS: error: invalid immediate expression
; CHECK-ERRORS: adds w3, w5, sym@PAGEOFF
; CHECK-ERRORS:              ^
; CHECK-ERRORS: error: invalid immediate expression
; CHECK-ERRORS: adds x9, x12, sym@PAGEOFF
; CHECK-ERRORS:               ^

sub x3, x5, sym@PAGEOFF
sub w20, w30, sym@PAGEOFF
; CHECK-ERRORS: error: invalid immediate expression
; CHECK-ERRORS: sub x3, x5, sym@PAGEOFF
; CHECK-ERRORS:             ^
; CHECK-ERRORS: error: invalid immediate expression
; CHECK-ERRORS: sub w20, w30, sym@PAGEOFF
; CHECK-ERRORS:               ^

subs w9, w10, sym@PAGEOFF
subs x20, x30, sym@PAGEOFF
; CHECK-ERRORS: error: invalid immediate expression
; CHECK-ERRORS: subs w9, w10, sym@PAGEOFF
; CHECK-ERRORS:               ^
; CHECK-ERRORS: error: invalid immediate expression
; CHECK-ERRORS: subs x20, x30, sym@PAGEOFF
; CHECK-ERRORS:                ^

tbl v0.8b, { v1 }, v0.8b
tbl v0.16b, { v1.8b, v2.8b, v3.8b }, v0.16b
tbx v3.16b, { v12.8b, v13.8b, v14.8b }, v6.8b
tbx v2.8b, { v0 }, v6.8b
; CHECK-ERRORS: error: invalid operand for instruction
; CHECK-ERRORS: tbl v0.8b, { v1 }, v0.8b
; CHECK-ERRORS:            ^
; CHECK-ERRORS: error: invalid operand for instruction
; CHECK-ERRORS: tbl v0.16b, { v1.8b, v2.8b, v3.8b }, v0.16b
; CHECK-ERRORS:             ^
; CHECK-ERRORS: error: invalid operand for instruction
; CHECK-ERRORS: tbx v3.16b, { v12.8b, v13.8b, v14.8b }, v6.8b
; CHECK-ERRORS:             ^
; CHECK-ERRORS: error: invalid operand for instruction
; CHECK-ERRORS: tbx v2.8b, { v0 }, v6.8b
; CHECK-ERRORS:            ^

b.c #0x4
; CHECK-ERRORS: error: invalid condition code
; CHECK-ERRORS: b.c #0x4
; CHECK-ERRORS:   ^

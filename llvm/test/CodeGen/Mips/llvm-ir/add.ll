; RUN: llc < %s -march=mips -mcpu=mips2 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=ALL-32BIT -check-prefix=M2
; RUN: llc < %s -march=mips -mcpu=mips32 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=ALL-32BIT -check-prefix=32R1-R2 -check-prefix=32R1
; RUN: llc < %s -march=mips -mcpu=mips32r2 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=ALL-32BIT -check-prefix=32R1-R2 -check-prefix=32R2
; RUN: llc < %s -march=mips -mcpu=mips32r6 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=ALL-32BIT -check-prefix=32R6
; RUN: llc < %s -march=mips64 -mcpu=mips4 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=ALL-64BIT -check-prefix=M4
; RUN: llc < %s -march=mips64 -mcpu=mips64 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=ALL-64BIT -check-prefix=64R1-R2
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=ALL-64BIT -check-prefix=64R1-R2
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 | FileCheck %s \
; RUN:     -check-prefix=ALL -check-prefix=ALL-64BIT -check-prefix=64R6

; FIXME: We should be able to have signext on the return value without incurring
;        a sign extend.
define i8 @add_i8(i8 signext %a, i8 signext %b) {
entry:
; ALL-LABEL: add_i8:

; ALL:           addu $2, $4, $5

  %r = add i8 %a, %b
  ret i8 %r
}

; FIXME: We should be able to have signext on the return value without incurring
;        a sign extend.
define i16 @add_i16(i16 signext %a, i16 signext %b) {
entry:
; ALL-LABEL: add_i16:

; ALL:           addu $2, $4, $5

  %r = add i16 %a, %b
  ret i16 %r
}

define signext i32 @add_i32(i32 signext %a, i32 signext %b) {
entry:
; ALL-LABEL: add_i32:

; ALL:           addu $2, $4, $5

  %r = add i32 %a, %b
  ret i32 %r
}

define signext i64 @add_i64(i64 signext %a, i64 signext %b) {
entry:
; ALL-LABEL: add_i64:

; ALL-32BIT-DAG: addu [[R1:\$3]], $5, $7
; ALL-32BIT-DAG: sltu [[T0:\$[0-9]+]], [[R1]], $7
; ALL-32BIT-DAG: addu [[T1:\$[0-9]+]], [[T0]], $6
; ALL-32BIT-DAG: addu $2, $4, [[T1]]

; ALL-64BIT:     daddu $2, $4, $5

  %r = add i64 %a, %b
  ret i64 %r
}

define signext i128 @add_i128(i128 signext %a, i128 signext %b) {
entry:
; ALL-LABEL: add_i128:

; ALL-32BIT-DAG: lw      [[A3:\$[0-9]+]], 28($sp)
; ALL-32BIT-DAG: addu    [[T0:\$[0-9]+]], $7, [[A3]]
; ALL-32BIT-DAG: sltu    [[T1:\$[0-9]+]], [[T0]], [[A3]]
; ALL-32BIT-DAG: lw      [[A2:\$[0-9]+]], 24($sp)
; ALL-32BIT-DAG: addu    [[T2:\$[0-9]+]], [[T1]], [[A2]]
; ALL-32BIT-DAG: addu    [[T3:\$[0-9]+]], $6, [[T2]]
; ALL-32BIT-DAG: sltu    [[T4:\$[0-9]+]], [[T3]], [[A2]]
; ALL-32BIT-DAG: lw      [[A1:\$[0-9]+]], 20($sp)
; ALL-32BIT-DAG: addu    [[T5:\$[0-9]+]], [[T4]], [[A1]]
; ALL-32BIT-DAG: lw      [[A0:\$[0-9]+]], 16($sp)
; ALL-32BIT-DAG: addu    [[R1:\$3]], [[R3:\$5]], [[T5]]
; ALL-32BIT-DAG: sltu    [[T6:\$[0-9]+]], [[R1]], [[A1]]
; ALL-32BIT-DAG: addu    [[T7:\$[0-9]+]], [[T6]], [[A0]]
; ALL-32BIT-DAG: addu    $2, [[R2:\$4]], [[T7]]
; ALL-32BIT-DAG: move    [[R2]], [[T3]]
; ALL-32BIT-DAG: move    [[R3]], [[T0]]

; ALL-64BIT-DAG: daddu [[R0:\$3]], $5, $7
; ALL-64BIT-DAG: sltu [[T1:\$[0-9]+]], [[R0]], $7
; ALL-64BIT-DAG: daddu [[T2:\$[0-9]+]], [[T1]], $6
; ALL-64BIT-DAG: daddu $2, $4, [[T2]]

  %r = add i128 %a, %b
  ret i128 %r
}

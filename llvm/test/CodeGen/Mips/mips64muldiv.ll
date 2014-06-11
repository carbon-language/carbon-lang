; RUN: llc -march=mips64el -mcpu=mips4 < %s | FileCheck %s -check-prefix=ALL
; RUN: llc -march=mips64el -mcpu=mips64 < %s | FileCheck %s -check-prefix=ALL

define i64 @m0(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: m0:
; ALL:           dmult ${{[45]}}, ${{[45]}}
; ALL:           mflo $2
  %mul = mul i64 %a1, %a0
  ret i64 %mul
}

define i64 @m1(i64 %a) nounwind readnone {
entry:
; ALL-LABEL: m1:
; ALL:           lui $[[T0:[0-9]+]], 21845
; ALL:           addiu $[[T0]], $[[T0]], 21845
; ALL:           dsll $[[T0]], $[[T0]], 16
; ALL:           addiu $[[T0]], $[[T0]], 21845
; ALL:           dsll $[[T0]], $[[T0]], 16
; ALL:           addiu $[[T0]], $[[T0]], 21846
; ALL:           dmult ${{[45]}}, $[[T0]]
; ALL:           mfhi $[[T1:[0-9]+]]
; ALL:           dsrl $2, $[[T1]], 63
; ALL:           daddu $2, $[[T1]], $2
  %div = sdiv i64 %a, 3
  ret i64 %div
}

define i64 @d0(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: d0:
; ALL:           ddivu $zero, $4, $5
; ALL:           mflo $2
  %div = udiv i64 %a0, %a1
  ret i64 %div
}

define i64 @d1(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: d1:
; ALL:           ddiv $zero, $4, $5
; ALL:           mflo $2
  %div = sdiv i64 %a0, %a1
  ret i64 %div
}

define i64 @d2(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: d2:
; ALL:           ddivu $zero, $4, $5
; ALL:           mfhi $2
  %rem = urem i64 %a0, %a1
  ret i64 %rem
}

define i64 @d3(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: d3:
; ALL:           ddiv $zero, $4, $5
; ALL:           mfhi $2
  %rem = srem i64 %a0, %a1
  ret i64 %rem
}

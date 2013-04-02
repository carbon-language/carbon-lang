; RUN: llc < %s -march=sparcv9 | FileCheck %s

; CHECK: ret2:
; CHECK: or %g0, %i1, %i0
define i64 @ret2(i64 %a, i64 %b) {
  ret i64 %b
}

; CHECK: shl_imm
; CHECK: sllx %i0, 7, %i0
define i64 @shl_imm(i64 %a) {
  %x = shl i64 %a, 7
  ret i64 %x
}

; CHECK: sra_reg
; CHECK: srax %i0, %i1, %i0
define i64 @sra_reg(i64 %a, i64 %b) {
  %x = ashr i64 %a, %b
  ret i64 %x
}

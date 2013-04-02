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

; Immediate materialization. Many of these patterns could actually be merged
; into the restore instruction:
;
;     restore %g0, %g0, %o0
;
; CHECK: ret_imm0
; CHECK: or %g0, %g0, %i0
define i64 @ret_imm0() {
  ret i64 0
}

; CHECK: ret_simm13
; CHECK: or %g0, -4096, %i0
define i64 @ret_simm13() {
  ret i64 -4096
}

; CHECK: ret_sethi
; CHECK: sethi 4, %i0
; CHECK-NOT: or
; CHECK: restore
define i64 @ret_sethi() {
  ret i64 4096
}

; CHECK: ret_sethi
; CHECK: sethi 4, [[R:%[goli][0-7]]]
; CHECK: or [[R]], 1, %i0
define i64 @ret_sethi_or() {
  ret i64 4097
}

; CHECK: ret_nimm33
; CHECK: sethi 4, [[R:%[goli][0-7]]]
; CHECK: xor [[R]], -4, %i0
define i64 @ret_nimm33() {
  ret i64 -4100
}

; CHECK: ret_bigimm
; CHECK: sethi
; CHECK: sethi
define i64 @ret_bigimm() {
  ret i64 6800754272627607872
}

; CHECK: reg_reg_alu
; CHECK: add %i0, %i1, [[R0:%[goli][0-7]]]
; CHECK: sub [[R0]], %i2, [[R1:%[goli][0-7]]]
; CHECK: andn [[R1]], %i0, %i0
define i64 @reg_reg_alu(i64 %x, i64 %y, i64 %z) {
  %a = add i64 %x, %y
  %b = sub i64 %a, %z
  %c = xor i64 %x, -1
  %d = and i64 %b, %c
  ret i64 %d
}

; CHECK: reg_imm_alu
; CHECK: add %i0, -5, [[R0:%[goli][0-7]]]
; CHECK: xor [[R0]], 2, %i0
define i64 @reg_imm_alu(i64 %x, i64 %y, i64 %z) {
  %a = add i64 %x, -5
  %b = xor i64 %a, 2
  ret i64 %b
}

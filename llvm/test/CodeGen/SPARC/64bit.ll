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

; CHECK: loads
; CHECK: ldx [%i0]
; CHECK: stx %
; CHECK: ld [%i1]
; CHECK: st %
; CHECK: ldsw [%i2]
; CHECK: stx %
; CHECK: ldsh [%i3]
; CHECK: sth %
define i64 @loads(i64* %p, i32* %q, i32* %r, i16* %s) {
  %a = load i64* %p
  %ai = add i64 1, %a
  store i64 %ai, i64* %p
  %b = load i32* %q
  %b2 = zext i32 %b to i64
  %bi = trunc i64 %ai to i32
  store i32 %bi, i32* %q
  %c = load i32* %r
  %c2 = sext i32 %c to i64
  store i64 %ai, i64* %p
  %d = load i16* %s
  %d2 = sext i16 %d to i64
  %di = trunc i64 %ai to i16
  store i16 %di, i16* %s

  %x1 = add i64 %a, %b2
  %x2 = add i64 %c2, %d2
  %x3 = add i64 %x1, %x2
  ret i64 %x3
}

; CHECK: stores
; CHECK: ldx [%i0+8], [[R:%[goli][0-7]]]
; CHECK: stx [[R]], [%i0+16]
; CHECK: st [[R]], [%i1+-8]
; CHECK: sth [[R]], [%i2+40]
; CHECK: stb [[R]], [%i3+-20]
define void @stores(i64* %p, i32* %q, i16* %r, i8* %s) {
  %p1 = getelementptr i64* %p, i64 1
  %p2 = getelementptr i64* %p, i64 2
  %pv = load i64* %p1
  store i64 %pv, i64* %p2

  %q2 = getelementptr i32* %q, i32 -2
  %qv = trunc i64 %pv to i32
  store i32 %qv, i32* %q2

  %r2 = getelementptr i16* %r, i16 20
  %rv = trunc i64 %pv to i16
  store i16 %rv, i16* %r2

  %s2 = getelementptr i8* %s, i8 -20
  %sv = trunc i64 %pv to i8
  store i8 %sv, i8* %s2

  ret void
}

; CHECK: promote_shifts
; CHECK: ldub [%i0], [[R:%[goli][0-7]]]
; CHECK: sll [[R]], [[R]], %i0
define i8 @promote_shifts(i8* %p) {
  %L24 = load i8* %p
  %L32 = load i8* %p
  %B36 = shl i8 %L24, %L32
  ret i8 %B36
}

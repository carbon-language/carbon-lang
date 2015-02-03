; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; Hexagon Programmer's Reference Manual 11.10.8 XTYPE/SHIFT

; Shift by immediate
declare i64 @llvm.hexagon.S2.asr.i.p(i64, i32)
define i64 @S2_asr_i_p(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.asr.i.p(i64 %a, i32 0)
  ret i64 %z
}
; CHECK: r1:0 = asr(r1:0, #0)

declare i64 @llvm.hexagon.S2.lsr.i.p(i64, i32)
define i64 @S2_lsr_i_p(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.lsr.i.p(i64 %a, i32 0)
  ret i64 %z
}
; CHECK: r1:0 = lsr(r1:0, #0)

declare i64 @llvm.hexagon.S2.asl.i.p(i64, i32)
define i64 @S2_asl_i_p(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.asl.i.p(i64 %a, i32 0)
  ret i64 %z
}
; CHECK: r1:0 = asl(r1:0, #0)

declare i32 @llvm.hexagon.S2.asr.i.r(i32, i32)
define i32 @S2_asr_i_r(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.asr.i.r(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = asr(r0, #0)

declare i32 @llvm.hexagon.S2.lsr.i.r(i32, i32)
define i32 @S2_lsr_i_r(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.lsr.i.r(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = lsr(r0, #0)

declare i32 @llvm.hexagon.S2.asl.i.r(i32, i32)
define i32 @S2_asl_i_r(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.asl.i.r(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = asl(r0, #0)

; Shift by immediate and accumulate
declare i64 @llvm.hexagon.S2.asr.i.p.nac(i64, i64, i32)
define i64 @S2_asr_i_p_nac(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.asr.i.p.nac(i64 %a, i64 %b, i32 0)
  ret i64 %z
}
; CHECK: r1:0 -= asr(r3:2, #0)

declare i64 @llvm.hexagon.S2.lsr.i.p.nac(i64, i64, i32)
define i64 @S2_lsr_i_p_nac(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.lsr.i.p.nac(i64 %a, i64 %b, i32 0)
  ret i64 %z
}
; CHECK: r1:0 -= lsr(r3:2, #0)

declare i64 @llvm.hexagon.S2.asl.i.p.nac(i64, i64, i32)
define i64 @S2_asl_i_p_nac(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.asl.i.p.nac(i64 %a, i64 %b, i32 0)
  ret i64 %z
}
; CHECK: r1:0 -= asl(r3:2, #0)

declare i64 @llvm.hexagon.S2.asr.i.p.acc(i64, i64, i32)
define i64 @S2_asr_i_p_acc(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.asr.i.p.acc(i64 %a, i64 %b, i32 0)
  ret i64 %z
}
; CHECK: r1:0 += asr(r3:2, #0)

declare i64 @llvm.hexagon.S2.lsr.i.p.acc(i64, i64, i32)
define i64 @S2_lsr_i_p_acc(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.lsr.i.p.acc(i64 %a, i64 %b, i32 0)
  ret i64 %z
}
; CHECK: r1:0 += lsr(r3:2, #0)

declare i64 @llvm.hexagon.S2.asl.i.p.acc(i64, i64, i32)
define i64 @S2_asl_i_p_acc(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.asl.i.p.acc(i64 %a, i64 %b, i32 0)
  ret i64 %z
}
; CHECK: r1:0 += asl(r3:2, #0)

declare i32 @llvm.hexagon.S2.asr.i.r.nac(i32, i32, i32)
define i32 @S2_asr_i_r_nac(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.asr.i.r.nac(i32 %a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 -= asr(r1, #0)

declare i32 @llvm.hexagon.S2.lsr.i.r.nac(i32, i32, i32)
define i32 @S2_lsr_i_r_nac(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.lsr.i.r.nac(i32 %a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 -= lsr(r1, #0)

declare i32 @llvm.hexagon.S2.asl.i.r.nac(i32, i32, i32)
define i32 @S2_asl_i_r_nac(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.asl.i.r.nac(i32 %a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 -= asl(r1, #0)

declare i32 @llvm.hexagon.S2.asr.i.r.acc(i32, i32, i32)
define i32 @S2_asr_i_r_acc(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.asr.i.r.acc(i32 %a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 += asr(r1, #0)

declare i32 @llvm.hexagon.S2.lsr.i.r.acc(i32, i32, i32)
define i32 @S2_lsr_i_r_acc(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.lsr.i.r.acc(i32 %a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 += lsr(r1, #0)

declare i32 @llvm.hexagon.S2.asl.i.r.acc(i32, i32, i32)
define i32 @S2_asl_i_r_acc(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.asl.i.r.acc(i32 %a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 += asl(r1, #0)

; Shift by immediate and add
declare i32 @llvm.hexagon.S4.addi.asl.ri(i32, i32, i32)
define i32 @S4_addi_asl_ri(i32 %a) {
  %z = call i32 @llvm.hexagon.S4.addi.asl.ri(i32 0, i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = add(#0, asl(r0, #0))

declare i32 @llvm.hexagon.S4.subi.asl.ri(i32, i32, i32)
define i32 @S4_subi_asl_ri(i32 %a) {
  %z = call i32 @llvm.hexagon.S4.subi.asl.ri(i32 0, i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = sub(#0, asl(r0, #0))

declare i32 @llvm.hexagon.S4.addi.lsr.ri(i32, i32, i32)
define i32 @S4_addi_lsr_ri(i32 %a) {
  %z = call i32 @llvm.hexagon.S4.addi.lsr.ri(i32 0, i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = add(#0, lsr(r0, #0))

declare i32 @llvm.hexagon.S4.subi.lsr.ri(i32, i32, i32)
define i32 @S4_subi_lsr_ri(i32 %a) {
  %z = call i32 @llvm.hexagon.S4.subi.lsr.ri(i32 0, i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = sub(#0, lsr(r0, #0))

declare i32 @llvm.hexagon.S2.addasl.rrri(i32, i32, i32)
define i32 @S2_addasl_rrri(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.addasl.rrri(i32 %a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 = addasl(r0, r1, #0)

; Shift by immediate and logical
declare i64 @llvm.hexagon.S2.asr.i.p.and(i64, i64, i32)
define i64 @S2_asr_i_p_and(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.asr.i.p.and(i64 %a, i64 %b, i32 0)
  ret i64 %z
}
; CHECK: r1:0 &= asr(r3:2, #0)

declare i64 @llvm.hexagon.S2.lsr.i.p.and(i64, i64, i32)
define i64 @S2_lsr_i_p_and(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.lsr.i.p.and(i64 %a, i64 %b, i32 0)
  ret i64 %z
}
; CHECK: r1:0 &= lsr(r3:2, #0)

declare i64 @llvm.hexagon.S2.asl.i.p.and(i64, i64, i32)
define i64 @S2_asl_i_p_and(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.asl.i.p.and(i64 %a, i64 %b, i32 0)
  ret i64 %z
}
; CHECK: r1:0 &= asl(r3:2, #0)

declare i64 @llvm.hexagon.S2.asr.i.p.or(i64, i64, i32)
define i64 @S2_asr_i_p_or(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.asr.i.p.or(i64 %a, i64 %b, i32 0)
  ret i64 %z
}
; CHECK: r1:0 |= asr(r3:2, #0)

declare i64 @llvm.hexagon.S2.lsr.i.p.or(i64, i64, i32)
define i64 @S2_lsr_i_p_or(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.lsr.i.p.or(i64 %a, i64 %b, i32 0)
  ret i64 %z
}
; CHECK: r1:0 |= lsr(r3:2, #0)

declare i64 @llvm.hexagon.S2.asl.i.p.or(i64, i64, i32)
define i64 @S2_asl_i_p_or(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.asl.i.p.or(i64 %a, i64 %b, i32 0)
  ret i64 %z
}
; CHECK: r1:0 |= asl(r3:2, #0)

declare i64 @llvm.hexagon.S2.lsr.i.p.xacc(i64, i64, i32)
define i64 @S2_lsr_i_p_xacc(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.lsr.i.p.xacc(i64 %a, i64 %b, i32 0)
  ret i64 %z
}
; CHECK: r1:0 ^= lsr(r3:2, #0)

declare i64 @llvm.hexagon.S2.asl.i.p.xacc(i64, i64, i32)
define i64 @S2_asl_i_p_xacc(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.asl.i.p.xacc(i64 %a, i64 %b, i32 0)
  ret i64 %z
}
; CHECK: r1:0 ^= asl(r3:2, #0)

declare i32 @llvm.hexagon.S2.asr.i.r.and(i32, i32, i32)
define i32 @S2_asr_i_r_and(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.asr.i.r.and(i32 %a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 &= asr(r1, #0)

declare i32 @llvm.hexagon.S2.lsr.i.r.and(i32, i32, i32)
define i32 @S2_lsr_i_r_and(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.lsr.i.r.and(i32 %a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 &= lsr(r1, #0)

declare i32 @llvm.hexagon.S2.asl.i.r.and(i32, i32, i32)
define i32 @S2_asl_i_r_and(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.asl.i.r.and(i32 %a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 &= asl(r1, #0)

declare i32 @llvm.hexagon.S2.asr.i.r.or(i32, i32, i32)
define i32 @S2_asr_i_r_or(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.asr.i.r.or(i32 %a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 |= asr(r1, #0)

declare i32 @llvm.hexagon.S2.lsr.i.r.or(i32, i32, i32)
define i32 @S2_lsr_i_r_or(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.lsr.i.r.or(i32 %a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 |= lsr(r1, #0)

declare i32 @llvm.hexagon.S2.asl.i.r.or(i32, i32, i32)
define i32 @S2_asl_i_r_or(i32%a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.asl.i.r.or(i32 %a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 |= asl(r1, #0)

declare i32 @llvm.hexagon.S2.lsr.i.r.xacc(i32, i32, i32)
define i32 @S2_lsr_i_r_xacc(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.lsr.i.r.xacc(i32%a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 ^= lsr(r1, #0)

declare i32 @llvm.hexagon.S2.asl.i.r.xacc(i32, i32, i32)
define i32 @S2_asl_i_r_xacc(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.asl.i.r.xacc(i32 %a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 ^= asl(r1, #0)

declare i32 @llvm.hexagon.S4.andi.asl.ri(i32, i32, i32)
define i32 @S4_andi_asl_ri(i32 %a) {
  %z = call i32 @llvm.hexagon.S4.andi.asl.ri(i32 0, i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = and(#0, asl(r0, #0))

declare i32 @llvm.hexagon.S4.ori.asl.ri(i32, i32, i32)
define i32 @S4_ori_asl_ri(i32 %a) {
  %z = call i32 @llvm.hexagon.S4.ori.asl.ri(i32 0, i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = or(#0, asl(r0, #0))

declare i32 @llvm.hexagon.S4.andi.lsr.ri(i32, i32, i32)
define i32 @S4_andi_lsr_ri(i32 %a) {
  %z = call i32 @llvm.hexagon.S4.andi.lsr.ri(i32 0, i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = and(#0, lsr(r0, #0))

declare i32 @llvm.hexagon.S4.ori.lsr.ri(i32, i32, i32)
define i32 @S4_ori_lsr_ri(i32 %a) {
  %z = call i32 @llvm.hexagon.S4.ori.lsr.ri(i32 0, i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = or(#0, lsr(r0, #0))

; Shift right by immediate with rounding
declare i64 @llvm.hexagon.S2.asr.i.p.rnd(i64, i32)
define i64 @S2_asr_i_p_rnd(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.asr.i.p.rnd(i64 %a, i32 0)
  ret i64 %z
}
; CHECK: r1:0 = asr(r1:0, #0):rnd

declare i32 @llvm.hexagon.S2.asr.i.r.rnd(i32, i32)
define i32 @S2_asr_i_r_rnd(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.asr.i.r.rnd(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = asr(r0, #0):rnd

; Shift left by immediate with saturation
declare i32 @llvm.hexagon.S2.asl.i.r.sat(i32, i32)
define i32 @S2_asl_i_r_sat(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.asl.i.r.sat(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = asl(r0, #0):sat

; Shift by register
declare i64 @llvm.hexagon.S2.asr.r.p(i64, i32)
define i64 @S2_asr_r_p(i64 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.S2.asr.r.p(i64 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = asr(r1:0, r2)

declare i64 @llvm.hexagon.S2.lsr.r.p(i64, i32)
define i64 @S2_lsr_r_p(i64 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.S2.lsr.r.p(i64 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = lsr(r1:0, r2)

declare i64 @llvm.hexagon.S2.asl.r.p(i64, i32)
define i64 @S2_asl_r_p(i64 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.S2.asl.r.p(i64 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = asl(r1:0, r2)

declare i64 @llvm.hexagon.S2.lsl.r.p(i64, i32)
define i64 @S2_lsl_r_p(i64 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.S2.lsl.r.p(i64 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = lsl(r1:0, r2)

declare i32 @llvm.hexagon.S2.asr.r.r(i32, i32)
define i32 @S2_asr_r_r(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.asr.r.r(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = asr(r0, r1)

declare i32 @llvm.hexagon.S2.lsr.r.r(i32, i32)
define i32 @S2_lsr_r_r(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.lsr.r.r(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = lsr(r0, r1)

declare i32 @llvm.hexagon.S2.asl.r.r(i32, i32)
define i32 @S2_asl_r_r(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.asl.r.r(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = asl(r0, r1)

declare i32 @llvm.hexagon.S2.lsl.r.r(i32, i32)
define i32 @S2_lsl_r_r(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.lsl.r.r(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = lsl(r0, r1)

declare i32 @llvm.hexagon.S4.lsli(i32, i32)
define i32 @S4_lsli(i32 %a) {
  %z = call i32 @llvm.hexagon.S4.lsli(i32 0, i32 %a)
  ret i32 %z
}
; CHECK: r0 = lsl(#0, r0)

; Shift by register and accumulate
declare i64 @llvm.hexagon.S2.asr.r.p.nac(i64, i64, i32)
define i64 @S2_asr_r_p_nac(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.asr.r.p.nac(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= asr(r3:2, r4)

declare i64 @llvm.hexagon.S2.lsr.r.p.nac(i64, i64, i32)
define i64 @S2_lsr_r_p_nac(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.lsr.r.p.nac(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= lsr(r3:2, r4)

declare i64 @llvm.hexagon.S2.asl.r.p.nac(i64, i64, i32)
define i64 @S2_asl_r_p_nac(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.asl.r.p.nac(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= asl(r3:2, r4)

declare i64 @llvm.hexagon.S2.lsl.r.p.nac(i64, i64, i32)
define i64 @S2_lsl_r_p_nac(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.lsl.r.p.nac(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= lsl(r3:2, r4)

declare i64 @llvm.hexagon.S2.asr.r.p.acc(i64, i64, i32)
define i64 @S2_asr_r_p_acc(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.asr.r.p.acc(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += asr(r3:2, r4)

declare i64 @llvm.hexagon.S2.lsr.r.p.acc(i64, i64, i32)
define i64 @S2_lsr_r_p_acc(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.lsr.r.p.acc(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += lsr(r3:2, r4)

declare i64 @llvm.hexagon.S2.asl.r.p.acc(i64, i64, i32)
define i64 @S2_asl_r_p_acc(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.asl.r.p.acc(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += asl(r3:2, r4)

declare i64 @llvm.hexagon.S2.lsl.r.p.acc(i64, i64, i32)
define i64 @S2_lsl_r_p_acc(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.lsl.r.p.acc(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += lsl(r3:2, r4)

declare i32 @llvm.hexagon.S2.asr.r.r.nac(i32, i32, i32)
define i32 @S2_asr_r_r_nac(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.S2.asr.r.r.nac(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= asr(r1, r2)

declare i32 @llvm.hexagon.S2.lsr.r.r.nac(i32, i32, i32)
define i32 @S2_lsr_r_r_nac(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.S2.lsr.r.r.nac(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= lsr(r1, r2)

declare i32 @llvm.hexagon.S2.asl.r.r.nac(i32, i32, i32)
define i32 @S2_asl_r_r_nac(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.S2.asl.r.r.nac(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= asl(r1, r2)

declare i32 @llvm.hexagon.S2.lsl.r.r.nac(i32, i32, i32)
define i32 @S2_lsl_r_r_nac(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.S2.lsl.r.r.nac(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= lsl(r1, r2)

declare i32 @llvm.hexagon.S2.asr.r.r.acc(i32, i32, i32)
define i32 @S2_asr_r_r_acc(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.S2.asr.r.r.acc(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += asr(r1, r2)

declare i32 @llvm.hexagon.S2.lsr.r.r.acc(i32, i32, i32)
define i32 @S2_lsr_r_r_acc(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.S2.lsr.r.r.acc(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += lsr(r1, r2)

declare i32 @llvm.hexagon.S2.asl.r.r.acc(i32, i32, i32)
define i32 @S2_asl_r_r_acc(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.S2.asl.r.r.acc(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += asl(r1, r2)

declare i32 @llvm.hexagon.S2.lsl.r.r.acc(i32, i32, i32)
define i32 @S2_lsl_r_r_acc(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.S2.lsl.r.r.acc(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += lsl(r1, r2)

; Shift by register and logical
declare i64 @llvm.hexagon.S2.asr.r.p.or(i64, i64, i32)
define i64 @S2_asr_r_p_or(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.asr.r.p.or(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 |= asr(r3:2, r4)

declare i64 @llvm.hexagon.S2.lsr.r.p.or(i64, i64, i32)
define i64 @S2_lsr_r_p_or(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.lsr.r.p.or(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 |= lsr(r3:2, r4)

declare i64 @llvm.hexagon.S2.asl.r.p.or(i64, i64, i32)
define i64 @S2_asl_r_p_or(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.asl.r.p.or(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 |= asl(r3:2, r4)

declare i64 @llvm.hexagon.S2.lsl.r.p.or(i64, i64, i32)
define i64 @S2_lsl_r_p_or(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.lsl.r.p.or(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 |= lsl(r3:2, r4)

declare i64 @llvm.hexagon.S2.asr.r.p.and(i64, i64, i32)
define i64 @S2_asr_r_p_and(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.asr.r.p.and(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 &= asr(r3:2, r4)

declare i64 @llvm.hexagon.S2.lsr.r.p.and(i64, i64, i32)
define i64 @S2_lsr_r_p_and(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.lsr.r.p.and(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 &= lsr(r3:2, r4)

declare i64 @llvm.hexagon.S2.asl.r.p.and(i64, i64, i32)
define i64 @S2_asl_r_p_and(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.asl.r.p.and(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 &= asl(r3:2, r4)

declare i64 @llvm.hexagon.S2.lsl.r.p.and(i64, i64, i32)
define i64 @S2_lsl_r_p_and(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.lsl.r.p.and(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 &= lsl(r3:2, r4)

declare i32 @llvm.hexagon.S2.asr.r.r.or(i32, i32, i32)
define i32 @S2_asr_r_r_or(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.S2.asr.r.r.or(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 |= asr(r1, r2)

declare i32 @llvm.hexagon.S2.lsr.r.r.or(i32, i32, i32)
define i32 @S2_lsr_r_r_or(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.S2.lsr.r.r.or(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 |= lsr(r1, r2)

declare i32 @llvm.hexagon.S2.asl.r.r.or(i32, i32, i32)
define i32 @S2_asl_r_r_or(i32%a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.S2.asl.r.r.or(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 |= asl(r1, r2)

declare i32 @llvm.hexagon.S2.lsl.r.r.or(i32, i32, i32)
define i32 @S2_lsl_r_r_or(i32%a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.S2.lsl.r.r.or(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 |= lsl(r1, r2)

declare i32 @llvm.hexagon.S2.asr.r.r.and(i32, i32, i32)
define i32 @S2_asr_r_r_and(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.S2.asr.r.r.and(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 &= asr(r1, r2)

declare i32 @llvm.hexagon.S2.lsr.r.r.and(i32, i32, i32)
define i32 @S2_lsr_r_r_and(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.S2.lsr.r.r.and(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 &= lsr(r1, r2)

declare i32 @llvm.hexagon.S2.asl.r.r.and(i32, i32, i32)
define i32 @S2_asl_r_r_and(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.S2.asl.r.r.and(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 &= asl(r1, r2)

declare i32 @llvm.hexagon.S2.lsl.r.r.and(i32, i32, i32)
define i32 @S2_lsl_r_r_and(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.S2.lsl.r.r.and(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 &= lsl(r1, r2)

; Shift by register with saturation
declare i32 @llvm.hexagon.S2.asr.r.r.sat(i32, i32)
define i32 @S2_asr_r_r_sat(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = asr(r0, r1):sat

declare i32 @llvm.hexagon.S2.asl.r.r.sat(i32, i32)
define i32 @S2_asl_r_r_sat(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.asl.r.r.sat(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = asl(r0, r1):sat

; Vector shift halfwords by immediate
declare i64 @llvm.hexagon.S2.asr.i.vh(i64, i32)
define i64 @S2_asr_i_vh(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.asr.i.vh(i64 %a, i32 0)
  ret i64 %z
}
; CHECK: r1:0 = vasrh(r1:0, #0)

declare i64 @llvm.hexagon.S2.lsr.i.vh(i64, i32)
define i64 @S2_lsr_i_vh(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.lsr.i.vh(i64 %a, i32 0)
  ret i64 %z
}
; CHECK: r1:0 = vlsrh(r1:0, #0)

declare i64 @llvm.hexagon.S2.asl.i.vh(i64, i32)
define i64 @S2_asl_i_vh(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.asl.i.vh(i64 %a, i32 0)
  ret i64 %z
}
; CHECK: r1:0 = vaslh(r1:0, #0)

; Vector shift halfwords by register
declare i64 @llvm.hexagon.S2.asr.r.vh(i64, i32)
define i64 @S2_asr_r_vh(i64 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.S2.asr.r.vh(i64 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = vasrh(r1:0, r2)

declare i64 @llvm.hexagon.S2.lsr.r.vh(i64, i32)
define i64 @S2_lsr_r_vh(i64 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.S2.lsr.r.vh(i64 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = vlsrh(r1:0, r2)

declare i64 @llvm.hexagon.S2.asl.r.vh(i64, i32)
define i64 @S2_asl_r_vh(i64 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.S2.asl.r.vh(i64 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = vaslh(r1:0, r2)

declare i64 @llvm.hexagon.S2.lsl.r.vh(i64, i32)
define i64 @S2_lsl_r_vh(i64 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.S2.lsl.r.vh(i64 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = vlslh(r1:0, r2)

; Vector shift words by immediate
declare i64 @llvm.hexagon.S2.asr.i.vw(i64, i32)
define i64 @S2_asr_i_vw(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.asr.i.vw(i64 %a, i32 0)
  ret i64 %z
}
; CHECK: r1:0 = vasrw(r1:0, #0)

declare i64 @llvm.hexagon.S2.lsr.i.vw(i64, i32)
define i64 @S2_lsr_i_vw(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.lsr.i.vw(i64 %a, i32 0)
  ret i64 %z
}
; CHECK: r1:0 = vlsrw(r1:0, #0)

declare i64 @llvm.hexagon.S2.asl.i.vw(i64, i32)
define i64 @S2_asl_i_vw(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.asl.i.vw(i64 %a, i32 0)
  ret i64 %z
}
; CHECK: r1:0 = vaslw(r1:0, #0)

; Vector shift words by with truncate and pack
declare i32 @llvm.hexagon.S2.asr.i.svw.trun(i64, i32)
define i32 @S2_asr_i_svw_trun(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.asr.i.svw.trun(i64 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = vasrw(r1:0, #0)

declare i32 @llvm.hexagon.S2.asr.r.svw.trun(i64, i32)
define i32 @S2_asr_r_svw_trun(i64 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.asr.r.svw.trun(i64 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = vasrw(r1:0, r2)

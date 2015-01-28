; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; Hexagon Programmer's Reference Manual 11.10.5 XTYPE/MPY

; Multiply and use lower result
declare i32 @llvm.hexagon.M4.mpyrr.addi(i32, i32, i32)
define i32 @M4_mpyrr_addi(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M4.mpyrr.addi(i32 0, i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = add(#0, mpyi(r0, r1))

declare i32 @llvm.hexagon.M4.mpyri.addi(i32, i32, i32)
define i32 @M4_mpyri_addi(i32 %a) {
  %z = call i32 @llvm.hexagon.M4.mpyri.addi(i32 0, i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = add(#0, mpyi(r0, #0))

declare i32 @llvm.hexagon.M4.mpyri.addr.u2(i32, i32, i32)
define i32 @M4_mpyri_addr_u2(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M4.mpyri.addr.u2(i32 %a, i32 0, i32 %b)
  ret i32 %z
}
; CHECK: r0 = add(r0, mpyi(#0, r1))

declare i32 @llvm.hexagon.M4.mpyri.addr(i32, i32, i32)
define i32 @M4_mpyri_addr(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M4.mpyri.addr(i32 %a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 = add(r0, mpyi(r1, #0))

declare i32 @llvm.hexagon.M4.mpyrr.addr(i32, i32, i32)
define i32 @M4_mpyrr_addr(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M4.mpyrr.addr(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r1 = add(r0, mpyi(r1, r2))

; Multiply signed halfwords
declare i64 @llvm.hexagon.M2.mpyd.ll.s0(i32, i32)
define i64 @M2_mpyd_ll_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyd.ll.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0.l, r1.l)

declare i64 @llvm.hexagon.M2.mpyd.ll.s1(i32, i32)
define i64 @M2_mpyd_ll_s1(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyd.ll.s1(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0.l, r1.l):<<1

declare i64 @llvm.hexagon.M2.mpyd.lh.s0(i32, i32)
define i64 @M2_mpyd_lh_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyd.lh.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0.l, r1.h)

declare i64 @llvm.hexagon.M2.mpyd.lh.s1(i32, i32)
define i64 @M2_mpyd_lh_s1(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyd.lh.s1(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0.l, r1.h):<<1

declare i64 @llvm.hexagon.M2.mpyd.hl.s0(i32, i32)
define i64 @M2_mpyd_hl_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyd.hl.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0.h, r1.l)

declare i64 @llvm.hexagon.M2.mpyd.hl.s1(i32, i32)
define i64 @M2_mpyd_hl_s1(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyd.hl.s1(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0.h, r1.l):<<1

declare i64 @llvm.hexagon.M2.mpyd.hh.s0(i32, i32)
define i64 @M2_mpyd_hh_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyd.hh.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0.h, r1.h)

declare i64 @llvm.hexagon.M2.mpyd.hh.s1(i32, i32)
define i64 @M2_mpyd_hh_s1(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyd.hh.s1(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0.h, r1.h):<<1

declare i64 @llvm.hexagon.M2.mpyd.rnd.ll.s0(i32, i32)
define i64 @M2_mpyd_rnd_ll_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyd.rnd.ll.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0.l, r1.l):rnd

declare i64 @llvm.hexagon.M2.mpyd.rnd.ll.s1(i32, i32)
define i64 @M2_mpyd_rnd_ll_s1(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyd.rnd.ll.s1(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0.l, r1.l):<<1:rnd

declare i64 @llvm.hexagon.M2.mpyd.rnd.lh.s0(i32, i32)
define i64 @M2_mpyd_rnd_lh_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyd.rnd.lh.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0.l, r1.h):rnd

declare i64 @llvm.hexagon.M2.mpyd.rnd.lh.s1(i32, i32)
define i64 @M2_mpyd_rnd_lh_s1(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyd.rnd.lh.s1(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0.l, r1.h):<<1:rnd

declare i64 @llvm.hexagon.M2.mpyd.rnd.hl.s0(i32, i32)
define i64 @M2_mpyd_rnd_hl_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyd.rnd.hl.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0.h, r1.l):rnd

declare i64 @llvm.hexagon.M2.mpyd.rnd.hl.s1(i32, i32)
define i64 @M2_mpyd_rnd_hl_s1(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyd.rnd.hl.s1(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0.h, r1.l):<<1:rnd

declare i64 @llvm.hexagon.M2.mpyd.rnd.hh.s0(i32, i32)
define i64 @M2_mpyd_rnd_hh_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyd.rnd.hh.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0.h, r1.h):rnd

declare i64 @llvm.hexagon.M2.mpyd.rnd.hh.s1(i32, i32)
define i64 @M2_mpyd_rnd_hh_s1(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyd.rnd.hh.s1(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0.h, r1.h):<<1:rnd

declare i64 @llvm.hexagon.M2.mpyd.acc.ll.s0(i64, i32, i32)
define i64 @M2_mpyd_acc_ll_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyd.acc.ll.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpy(r2.l, r3.l)

declare i64 @llvm.hexagon.M2.mpyd.acc.ll.s1(i64, i32, i32)
define i64 @M2_mpyd_acc_ll_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyd.acc.ll.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpy(r2.l, r3.l):<<1

declare i64 @llvm.hexagon.M2.mpyd.acc.lh.s0(i64, i32, i32)
define i64 @M2_mpyd_acc_lh_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyd.acc.lh.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpy(r2.l, r3.h)

declare i64 @llvm.hexagon.M2.mpyd.acc.lh.s1(i64, i32, i32)
define i64 @M2_mpyd_acc_lh_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyd.acc.lh.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpy(r2.l, r3.h):<<1

declare i64 @llvm.hexagon.M2.mpyd.acc.hl.s0(i64, i32, i32)
define i64 @M2_mpyd_acc_hl_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyd.acc.hl.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpy(r2.h, r3.l)

declare i64 @llvm.hexagon.M2.mpyd.acc.hl.s1(i64, i32, i32)
define i64 @M2_mpyd_acc_hl_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyd.acc.hl.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpy(r2.h, r3.l):<<1

declare i64 @llvm.hexagon.M2.mpyd.acc.hh.s0(i64, i32, i32)
define i64 @M2_mpyd_acc_hh_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyd.acc.hh.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpy(r2.h, r3.h)

declare i64 @llvm.hexagon.M2.mpyd.acc.hh.s1(i64, i32, i32)
define i64 @M2_mpyd_acc_hh_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyd.acc.hh.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpy(r2.h, r3.h):<<1

declare i64 @llvm.hexagon.M2.mpyd.nac.ll.s0(i64, i32, i32)
define i64 @M2_mpyd_nac_ll_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyd.nac.ll.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpy(r2.l, r3.l)

declare i64 @llvm.hexagon.M2.mpyd.nac.ll.s1(i64, i32, i32)
define i64 @M2_mpyd_nac_ll_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyd.nac.ll.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpy(r2.l, r3.l):<<1

declare i64 @llvm.hexagon.M2.mpyd.nac.lh.s0(i64, i32, i32)
define i64 @M2_mpyd_nac_lh_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyd.nac.lh.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpy(r2.l, r3.h)

declare i64 @llvm.hexagon.M2.mpyd.nac.lh.s1(i64, i32, i32)
define i64 @M2_mpyd_nac_lh_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyd.nac.lh.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpy(r2.l, r3.h):<<1

declare i64 @llvm.hexagon.M2.mpyd.nac.hl.s0(i64, i32, i32)
define i64 @M2_mpyd_nac_hl_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyd.nac.hl.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpy(r2.h, r3.l)

declare i64 @llvm.hexagon.M2.mpyd.nac.hl.s1(i64, i32, i32)
define i64 @M2_mpyd_nac_hl_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyd.nac.hl.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpy(r2.h, r3.l):<<1

declare i64 @llvm.hexagon.M2.mpyd.nac.hh.s0(i64, i32, i32)
define i64 @M2_mpyd_nac_hh_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyd.nac.hh.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpy(r2.h, r3.h)

declare i64 @llvm.hexagon.M2.mpyd.nac.hh.s1(i64, i32, i32)
define i64 @M2_mpyd_nac_hh_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyd.nac.hh.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpy(r2.h, r3.h):<<1

declare i32 @llvm.hexagon.M2.mpy.ll.s0(i32, i32)
define i32 @M2_mpy_ll_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.ll.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.l, r1.l)

declare i32 @llvm.hexagon.M2.mpy.ll.s1(i32, i32)
define i32 @M2_mpy_ll_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.ll.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.l, r1.l):<<1

declare i32 @llvm.hexagon.M2.mpy.lh.s0(i32, i32)
define i32 @M2_mpy_lh_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.lh.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.l, r1.h)

declare i32 @llvm.hexagon.M2.mpy.lh.s1(i32, i32)
define i32 @M2_mpy_lh_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.lh.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.l, r1.h):<<1

declare i32 @llvm.hexagon.M2.mpy.hl.s0(i32, i32)
define i32 @M2_mpy_hl_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.hl.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.h, r1.l)

declare i32 @llvm.hexagon.M2.mpy.hl.s1(i32, i32)
define i32 @M2_mpy_hl_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.hl.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.h, r1.l):<<1

declare i32 @llvm.hexagon.M2.mpy.hh.s0(i32, i32)
define i32 @M2_mpy_hh_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.hh.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.h, r1.h)

declare i32 @llvm.hexagon.M2.mpy.hh.s1(i32, i32)
define i32 @M2_mpy_hh_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.hh.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.h, r1.h):<<1

declare i32 @llvm.hexagon.M2.mpy.sat.ll.s0(i32, i32)
define i32 @M2_mpy_sat_ll_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.sat.ll.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.l, r1.l):sat

declare i32 @llvm.hexagon.M2.mpy.sat.ll.s1(i32, i32)
define i32 @M2_mpy_sat_ll_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.sat.ll.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.l, r1.l):<<1:sat

declare i32 @llvm.hexagon.M2.mpy.sat.lh.s0(i32, i32)
define i32 @M2_mpy_sat_lh_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.sat.lh.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.l, r1.h):sat

declare i32 @llvm.hexagon.M2.mpy.sat.lh.s1(i32, i32)
define i32 @M2_mpy_sat_lh_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.sat.lh.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.l, r1.h):<<1:sat

declare i32 @llvm.hexagon.M2.mpy.sat.hl.s0(i32, i32)
define i32 @M2_mpy_sat_hl_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.sat.hl.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.h, r1.l):sat

declare i32 @llvm.hexagon.M2.mpy.sat.hl.s1(i32, i32)
define i32 @M2_mpy_sat_hl_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.sat.hl.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.h, r1.l):<<1:sat

declare i32 @llvm.hexagon.M2.mpy.sat.hh.s0(i32, i32)
define i32 @M2_mpy_sat_hh_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.sat.hh.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.h, r1.h):sat

declare i32 @llvm.hexagon.M2.mpy.sat.hh.s1(i32, i32)
define i32 @M2_mpy_sat_hh_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.sat.hh.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.h, r1.h):<<1:sat

declare i32 @llvm.hexagon.M2.mpy.sat.rnd.ll.s0(i32, i32)
define i32 @M2_mpy_sat_rnd_ll_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.sat.rnd.ll.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.l, r1.l):rnd:sat

declare i32 @llvm.hexagon.M2.mpy.sat.rnd.ll.s1(i32, i32)
define i32 @M2_mpy_sat_rnd_ll_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.sat.rnd.ll.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.l, r1.l):<<1:rnd:sat

declare i32 @llvm.hexagon.M2.mpy.sat.rnd.lh.s0(i32, i32)
define i32 @M2_mpy_sat_rnd_lh_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.sat.rnd.lh.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.l, r1.h):rnd:sat

declare i32 @llvm.hexagon.M2.mpy.sat.rnd.lh.s1(i32, i32)
define i32 @M2_mpy_sat_rnd_lh_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.sat.rnd.lh.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.l, r1.h):<<1:rnd:sat

declare i32 @llvm.hexagon.M2.mpy.sat.rnd.hl.s0(i32, i32)
define i32 @M2_mpy_sat_rnd_hl_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.sat.rnd.hl.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.h, r1.l):rnd:sat

declare i32 @llvm.hexagon.M2.mpy.sat.rnd.hl.s1(i32, i32)
define i32 @M2_mpy_sat_rnd_hl_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.sat.rnd.hl.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.h, r1.l):<<1:rnd:sat

declare i32 @llvm.hexagon.M2.mpy.sat.rnd.hh.s0(i32, i32)
define i32 @M2_mpy_sat_rnd_hh_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.sat.rnd.hh.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.h, r1.h):rnd:sat

declare i32 @llvm.hexagon.M2.mpy.sat.rnd.hh.s1(i32, i32)
define i32 @M2_mpy_sat_rnd_hh_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.sat.rnd.hh.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0.h, r1.h):<<1:rnd:sat

declare i32 @llvm.hexagon.M2.mpy.acc.ll.s0(i32, i32, i32)
define i32 @M2_mpy_acc_ll_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.acc.ll.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1.l, r2.l)

declare i32 @llvm.hexagon.M2.mpy.acc.ll.s1(i32, i32, i32)
define i32 @M2_mpy_acc_ll_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.acc.ll.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1.l, r2.l):<<1

declare i32 @llvm.hexagon.M2.mpy.acc.lh.s0(i32, i32, i32)
define i32 @M2_mpy_acc_lh_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.acc.lh.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1.l, r2.h)

declare i32 @llvm.hexagon.M2.mpy.acc.lh.s1(i32, i32, i32)
define i32 @M2_mpy_acc_lh_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.acc.lh.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1.l, r2.h):<<1

declare i32 @llvm.hexagon.M2.mpy.acc.hl.s0(i32, i32, i32)
define i32 @M2_mpy_acc_hl_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.acc.hl.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1.h, r2.l)

declare i32 @llvm.hexagon.M2.mpy.acc.hl.s1(i32, i32, i32)
define i32 @M2_mpy_acc_hl_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.acc.hl.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1.h, r2.l):<<1

declare i32 @llvm.hexagon.M2.mpy.acc.hh.s0(i32, i32, i32)
define i32 @M2_mpy_acc_hh_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.acc.hh.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1.h, r2.h)

declare i32 @llvm.hexagon.M2.mpy.acc.hh.s1(i32, i32, i32)
define i32 @M2_mpy_acc_hh_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.acc.hh.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1.h, r2.h):<<1

declare i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s0(i32, i32, i32)
define i32 @M2_mpy_acc_sat_ll_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1.l, r2.l):sat

declare i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32, i32, i32)
define i32 @M2_mpy_acc_sat_ll_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1.l, r2.l):<<1:sat

declare i32 @llvm.hexagon.M2.mpy.acc.sat.lh.s0(i32, i32, i32)
define i32 @M2_mpy_acc_sat_lh_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.acc.sat.lh.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1.l, r2.h):sat

declare i32 @llvm.hexagon.M2.mpy.acc.sat.lh.s1(i32, i32, i32)
define i32 @M2_mpy_acc_sat_lh_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.acc.sat.lh.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1.l, r2.h):<<1:sat

declare i32 @llvm.hexagon.M2.mpy.acc.sat.hl.s0(i32, i32, i32)
define i32 @M2_mpy_acc_sat_hl_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.acc.sat.hl.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1.h, r2.l):sat

declare i32 @llvm.hexagon.M2.mpy.acc.sat.hl.s1(i32, i32, i32)
define i32 @M2_mpy_acc_sat_hl_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.acc.sat.hl.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1.h, r2.l):<<1:sat

declare i32 @llvm.hexagon.M2.mpy.acc.sat.hh.s0(i32, i32, i32)
define i32 @M2_mpy_acc_sat_hh_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.acc.sat.hh.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1.h, r2.h):sat

declare i32 @llvm.hexagon.M2.mpy.acc.sat.hh.s1(i32, i32, i32)
define i32 @M2_mpy_acc_sat_hh_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.acc.sat.hh.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1.h, r2.h):<<1:sat

declare i32 @llvm.hexagon.M2.mpy.nac.ll.s0(i32, i32, i32)
define i32 @M2_mpy_nac_ll_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.nac.ll.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1.l, r2.l)

declare i32 @llvm.hexagon.M2.mpy.nac.ll.s1(i32, i32, i32)
define i32 @M2_mpy_nac_ll_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.nac.ll.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1.l, r2.l):<<1

declare i32 @llvm.hexagon.M2.mpy.nac.lh.s0(i32, i32, i32)
define i32 @M2_mpy_nac_lh_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.nac.lh.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1.l, r2.h)

declare i32 @llvm.hexagon.M2.mpy.nac.lh.s1(i32, i32, i32)
define i32 @M2_mpy_nac_lh_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.nac.lh.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1.l, r2.h):<<1

declare i32 @llvm.hexagon.M2.mpy.nac.hl.s0(i32, i32, i32)
define i32 @M2_mpy_nac_hl_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.nac.hl.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1.h, r2.l)

declare i32 @llvm.hexagon.M2.mpy.nac.hl.s1(i32, i32, i32)
define i32 @M2_mpy_nac_hl_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.nac.hl.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1.h, r2.l):<<1

declare i32 @llvm.hexagon.M2.mpy.nac.hh.s0(i32, i32, i32)
define i32 @M2_mpy_nac_hh_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.nac.hh.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1.h, r2.h)

declare i32 @llvm.hexagon.M2.mpy.nac.hh.s1(i32, i32, i32)
define i32 @M2_mpy_nac_hh_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.nac.hh.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1.h, r2.h):<<1

declare i32 @llvm.hexagon.M2.mpy.nac.sat.ll.s0(i32, i32, i32)
define i32 @M2_mpy_nac_sat_ll_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.nac.sat.ll.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1.l, r2.l):sat

declare i32 @llvm.hexagon.M2.mpy.nac.sat.ll.s1(i32, i32, i32)
define i32 @M2_mpy_nac_sat_ll_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.nac.sat.ll.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1.l, r2.l):<<1:sat

declare i32 @llvm.hexagon.M2.mpy.nac.sat.lh.s0(i32, i32, i32)
define i32 @M2_mpy_nac_sat_lh_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.nac.sat.lh.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1.l, r2.h):sat

declare i32 @llvm.hexagon.M2.mpy.nac.sat.lh.s1(i32, i32, i32)
define i32 @M2_mpy_nac_sat_lh_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.nac.sat.lh.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1.l, r2.h):<<1:sat

declare i32 @llvm.hexagon.M2.mpy.nac.sat.hl.s0(i32, i32, i32)
define i32 @M2_mpy_nac_sat_hl_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.nac.sat.hl.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1.h, r2.l):sat

declare i32 @llvm.hexagon.M2.mpy.nac.sat.hl.s1(i32, i32, i32)
define i32 @M2_mpy_nac_sat_hl_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.nac.sat.hl.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1.h, r2.l):<<1:sat

declare i32 @llvm.hexagon.M2.mpy.nac.sat.hh.s0(i32, i32, i32)
define i32 @M2_mpy_nac_sat_hh_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.nac.sat.hh.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1.h, r2.h):sat

declare i32 @llvm.hexagon.M2.mpy.nac.sat.hh.s1(i32, i32, i32)
define i32 @M2_mpy_nac_sat_hh_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpy.nac.sat.hh.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1.h, r2.h):<<1:sat

; Multiply unsigned halfwords
declare i64 @llvm.hexagon.M2.mpyud.ll.s0(i32, i32)
define i64 @M2_mpyud_ll_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyud.ll.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpyu(r0.l, r1.l)

declare i64 @llvm.hexagon.M2.mpyud.ll.s1(i32, i32)
define i64 @M2_mpyud_ll_s1(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyud.ll.s1(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpyu(r0.l, r1.l):<<1

declare i64 @llvm.hexagon.M2.mpyud.lh.s0(i32, i32)
define i64 @M2_mpyud_lh_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyud.lh.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpyu(r0.l, r1.h)

declare i64 @llvm.hexagon.M2.mpyud.lh.s1(i32, i32)
define i64 @M2_mpyud_lh_s1(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyud.lh.s1(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpyu(r0.l, r1.h):<<1

declare i64 @llvm.hexagon.M2.mpyud.hl.s0(i32, i32)
define i64 @M2_mpyud_hl_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyud.hl.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpyu(r0.h, r1.l)

declare i64 @llvm.hexagon.M2.mpyud.hl.s1(i32, i32)
define i64 @M2_mpyud_hl_s1(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyud.hl.s1(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpyu(r0.h, r1.l):<<1

declare i64 @llvm.hexagon.M2.mpyud.hh.s0(i32, i32)
define i64 @M2_mpyud_hh_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyud.hh.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpyu(r0.h, r1.h)

declare i64 @llvm.hexagon.M2.mpyud.hh.s1(i32, i32)
define i64 @M2_mpyud_hh_s1(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.mpyud.hh.s1(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpyu(r0.h, r1.h):<<1

declare i64 @llvm.hexagon.M2.mpyud.acc.ll.s0(i64, i32, i32)
define i64 @M2_mpyud_acc_ll_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyud.acc.ll.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpyu(r2.l, r3.l)

declare i64 @llvm.hexagon.M2.mpyud.acc.ll.s1(i64, i32, i32)
define i64 @M2_mpyud_acc_ll_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyud.acc.ll.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpyu(r2.l, r3.l):<<1

declare i64 @llvm.hexagon.M2.mpyud.acc.lh.s0(i64, i32, i32)
define i64 @M2_mpyud_acc_lh_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyud.acc.lh.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpyu(r2.l, r3.h)

declare i64 @llvm.hexagon.M2.mpyud.acc.lh.s1(i64, i32, i32)
define i64 @M2_mpyud_acc_lh_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyud.acc.lh.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpyu(r2.l, r3.h):<<1

declare i64 @llvm.hexagon.M2.mpyud.acc.hl.s0(i64, i32, i32)
define i64 @M2_mpyud_acc_hl_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyud.acc.hl.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpyu(r2.h, r3.l)

declare i64 @llvm.hexagon.M2.mpyud.acc.hl.s1(i64, i32, i32)
define i64 @M2_mpyud_acc_hl_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyud.acc.hl.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpyu(r2.h, r3.l):<<1

declare i64 @llvm.hexagon.M2.mpyud.acc.hh.s0(i64, i32, i32)
define i64 @M2_mpyud_acc_hh_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyud.acc.hh.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpyu(r2.h, r3.h)

declare i64 @llvm.hexagon.M2.mpyud.acc.hh.s1(i64, i32, i32)
define i64 @M2_mpyud_acc_hh_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyud.acc.hh.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpyu(r2.h, r3.h):<<1

declare i64 @llvm.hexagon.M2.mpyud.nac.ll.s0(i64, i32, i32)
define i64 @M2_mpyud_nac_ll_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyud.nac.ll.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpyu(r2.l, r3.l)

declare i64 @llvm.hexagon.M2.mpyud.nac.ll.s1(i64, i32, i32)
define i64 @M2_mpyud_nac_ll_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyud.nac.ll.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpyu(r2.l, r3.l):<<1

declare i64 @llvm.hexagon.M2.mpyud.nac.lh.s0(i64, i32, i32)
define i64 @M2_mpyud_nac_lh_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyud.nac.lh.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpyu(r2.l, r3.h)

declare i64 @llvm.hexagon.M2.mpyud.nac.lh.s1(i64, i32, i32)
define i64 @M2_mpyud_nac_lh_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyud.nac.lh.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpyu(r2.l, r3.h):<<1

declare i64 @llvm.hexagon.M2.mpyud.nac.hl.s0(i64, i32, i32)
define i64 @M2_mpyud_nac_hl_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyud.nac.hl.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpyu(r2.h, r3.l)

declare i64 @llvm.hexagon.M2.mpyud.nac.hl.s1(i64, i32, i32)
define i64 @M2_mpyud_nac_hl_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyud.nac.hl.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpyu(r2.h, r3.l):<<1

declare i64 @llvm.hexagon.M2.mpyud.nac.hh.s0(i64, i32, i32)
define i64 @M2_mpyud_nac_hh_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyud.nac.hh.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpyu(r2.h, r3.h)

declare i64 @llvm.hexagon.M2.mpyud.nac.hh.s1(i64, i32, i32)
define i64 @M2_mpyud_nac_hh_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.mpyud.nac.hh.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpyu(r2.h, r3.h):<<1

declare i32 @llvm.hexagon.M2.mpyu.ll.s0(i32, i32)
define i32 @M2_mpyu_ll_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpyu.ll.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpyu(r0.l, r1.l)

declare i32 @llvm.hexagon.M2.mpyu.ll.s1(i32, i32)
define i32 @M2_mpyu_ll_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpyu.ll.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpyu(r0.l, r1.l):<<1

declare i32 @llvm.hexagon.M2.mpyu.lh.s0(i32, i32)
define i32 @M2_mpyu_lh_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpyu.lh.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpyu(r0.l, r1.h)

declare i32 @llvm.hexagon.M2.mpyu.lh.s1(i32, i32)
define i32 @M2_mpyu_lh_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpyu.lh.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpyu(r0.l, r1.h):<<1

declare i32 @llvm.hexagon.M2.mpyu.hl.s0(i32, i32)
define i32 @M2_mpyu_hl_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpyu.hl.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpyu(r0.h, r1.l)

declare i32 @llvm.hexagon.M2.mpyu.hl.s1(i32, i32)
define i32 @M2_mpyu_hl_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpyu.hl.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpyu(r0.h, r1.l):<<1

declare i32 @llvm.hexagon.M2.mpyu.hh.s0(i32, i32)
define i32 @M2_mpyu_hh_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpyu.hh.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpyu(r0.h, r1.h)

declare i32 @llvm.hexagon.M2.mpyu.hh.s1(i32, i32)
define i32 @M2_mpyu_hh_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpyu.hh.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpyu(r0.h, r1.h):<<1

declare i32 @llvm.hexagon.M2.mpyu.acc.ll.s0(i32, i32, i32)
define i32 @M2_mpyu_acc_ll_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpyu.acc.ll.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpyu(r1.l, r2.l)

declare i32 @llvm.hexagon.M2.mpyu.acc.ll.s1(i32, i32, i32)
define i32 @M2_mpyu_acc_ll_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpyu.acc.ll.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpyu(r1.l, r2.l):<<1

declare i32 @llvm.hexagon.M2.mpyu.acc.lh.s0(i32, i32, i32)
define i32 @M2_mpyu_acc_lh_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpyu.acc.lh.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpyu(r1.l, r2.h)

declare i32 @llvm.hexagon.M2.mpyu.acc.lh.s1(i32, i32, i32)
define i32 @M2_mpyu_acc_lh_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpyu.acc.lh.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpyu(r1.l, r2.h):<<1

declare i32 @llvm.hexagon.M2.mpyu.acc.hl.s0(i32, i32, i32)
define i32 @M2_mpyu_acc_hl_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpyu.acc.hl.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpyu(r1.h, r2.l)

declare i32 @llvm.hexagon.M2.mpyu.acc.hl.s1(i32, i32, i32)
define i32 @M2_mpyu_acc_hl_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpyu.acc.hl.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpyu(r1.h, r2.l):<<1

declare i32 @llvm.hexagon.M2.mpyu.acc.hh.s0(i32, i32, i32)
define i32 @M2_mpyu_acc_hh_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpyu.acc.hh.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpyu(r1.h, r2.h)

declare i32 @llvm.hexagon.M2.mpyu.acc.hh.s1(i32, i32, i32)
define i32 @M2_mpyu_acc_hh_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpyu.acc.hh.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpyu(r1.h, r2.h):<<1

declare i32 @llvm.hexagon.M2.mpyu.nac.ll.s0(i32, i32, i32)
define i32 @M2_mpyu_nac_ll_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpyu.nac.ll.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpyu(r1.l, r2.l)

declare i32 @llvm.hexagon.M2.mpyu.nac.ll.s1(i32, i32, i32)
define i32 @M2_mpyu_nac_ll_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpyu.nac.ll.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpyu(r1.l, r2.l):<<1

declare i32 @llvm.hexagon.M2.mpyu.nac.lh.s0(i32, i32, i32)
define i32 @M2_mpyu_nac_lh_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpyu.nac.lh.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpyu(r1.l, r2.h)

declare i32 @llvm.hexagon.M2.mpyu.nac.lh.s1(i32, i32, i32)
define i32 @M2_mpyu_nac_lh_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpyu.nac.lh.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpyu(r1.l, r2.h):<<1

declare i32 @llvm.hexagon.M2.mpyu.nac.hl.s0(i32, i32, i32)
define i32 @M2_mpyu_nac_hl_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpyu.nac.hl.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpyu(r1.h, r2.l)

declare i32 @llvm.hexagon.M2.mpyu.nac.hl.s1(i32, i32, i32)
define i32 @M2_mpyu_nac_hl_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpyu.nac.hl.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpyu(r1.h, r2.l):<<1

declare i32 @llvm.hexagon.M2.mpyu.nac.hh.s0(i32, i32, i32)
define i32 @M2_mpyu_nac_hh_s0(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpyu.nac.hh.s0(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpyu(r1.h, r2.h)

declare i32 @llvm.hexagon.M2.mpyu.nac.hh.s1(i32, i32, i32)
define i32 @M2_mpyu_nac_hh_s1(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M2.mpyu.nac.hh.s1(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpyu(r1.h, r2.h):<<1

; Polynomial multiply words
declare i64 @llvm.hexagon.M4.pmpyw(i32, i32)
define i64 @M4_pmpyw(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M4.pmpyw(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = pmpyw(r0, r1)

declare i64 @llvm.hexagon.M4.pmpyw.acc(i64, i32, i32)
define i64 @M4_pmpyw_acc(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M4.pmpyw.acc(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 ^= pmpyw(r2, r3)

; Multiply and use upper result
declare i32 @llvm.hexagon.M2.dpmpyss.rnd.s0(i32, i32)
define i32 @M2_dpmpyss_rnd_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.dpmpyss.rnd.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0, r1):rnd

declare i32 @llvm.hexagon.M2.mpyu.up(i32, i32)
define i32 @M2_mpyu_up(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpyu.up(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpyu(r0, r1)

declare i32 @llvm.hexagon.M2.mpysu.up(i32, i32)
define i32 @M2_mpysu_up(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpysu.up(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpysu(r0, r1)

declare i32 @llvm.hexagon.M2.hmmpyh.s1(i32, i32)
define i32 @M2_hmmpyh_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.hmmpyh.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0, r1.h):<<1:sat

declare i32 @llvm.hexagon.M2.hmmpyl.s1(i32, i32)
define i32 @M2_hmmpyl_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.hmmpyl.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0, r1.l):<<1:sat

declare i32 @llvm.hexagon.M2.hmmpyh.rs1(i32, i32)
define i32 @M2_hmmpyh_rs1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.hmmpyh.rs1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0, r1.h):<<1:rnd:sat

declare i32 @llvm.hexagon.M2.mpy.up.s1.sat(i32, i32)
define i32 @M2_mpy_up_s1_sat(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.up.s1.sat(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0, r1):<<1:sat

declare i32 @llvm.hexagon.M2.hmmpyl.rs1(i32, i32)
define i32 @M2_hmmpyl_rs1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.hmmpyl.rs1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0, r1.l):<<1:rnd:sat

declare i32 @llvm.hexagon.M2.mpy.up(i32, i32)
define i32 @M2_mpy_up(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.up(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0, r1)

declare i32 @llvm.hexagon.M2.mpy.up.s1(i32, i32)
define i32 @M2_mpy_up_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.mpy.up.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mpy(r0, r1):<<1

declare i32 @llvm.hexagon.M4.mac.up.s1.sat(i32, i32, i32)
define i32 @M4_mac_up_s1_sat(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M4.mac.up.s1.sat(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 += mpy(r1, r2):<<1:sat

declare i32 @llvm.hexagon.M4.nac.up.s1.sat(i32, i32, i32)
define i32 @M4_nac_up_s1_sat(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.M4.nac.up.s1.sat(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 -= mpy(r1, r2):<<1:sat

; Multiply and use full result
declare i64 @llvm.hexagon.M2.dpmpyss.s0(i32, i32)
define i64 @M2_dpmpyss_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.dpmpyss.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpy(r0, r1)

declare i64 @llvm.hexagon.M2.dpmpyuu.s0(i32, i32)
define i64 @M2_dpmpyuu_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.dpmpyuu.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = mpyu(r0, r1)

declare i64 @llvm.hexagon.M2.dpmpyss.acc.s0(i64, i32, i32)
define i64 @M2_dpmpyss_acc_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.dpmpyss.acc.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpy(r2, r3)

declare i64 @llvm.hexagon.M2.dpmpyss.nac.s0(i64, i32, i32)
define i64 @M2_dpmpyss_nac_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.dpmpyss.nac.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpy(r2, r3)

declare i64 @llvm.hexagon.M2.dpmpyuu.acc.s0(i64, i32, i32)
define i64 @M2_dpmpyuu_acc_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.dpmpyuu.acc.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 += mpyu(r2, r3)

declare i64 @llvm.hexagon.M2.dpmpyuu.nac.s0(i64, i32, i32)
define i64 @M2_dpmpyuu_nac_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.dpmpyuu.nac.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 -= mpyu(r2, r3)

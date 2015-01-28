; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; Hexagon Programmer's Reference Manual 11.10.2 XTYPE/BIT

; Count leading
declare i32 @llvm.hexagon.S2.clbp(i64)
define i32 @S2_clbp(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.clbp(i64 %a)
  ret i32 %z
}
; CHECK: r0 = clb(r1:0)

declare i32 @llvm.hexagon.S2.cl0p(i64)
define i32 @S2_cl0p(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.cl0p(i64 %a)
  ret i32 %z
}
; CHECK: r0 = cl0(r1:0)

declare i32 @llvm.hexagon.S2.cl1p(i64)
define i32 @S2_cl1p(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.cl1p(i64 %a)
  ret i32 %z
}
; CHECK: r0 = cl1(r1:0)

declare i32 @llvm.hexagon.S4.clbpnorm(i64)
define i32 @S4_clbpnorm(i64 %a) {
  %z = call i32 @llvm.hexagon.S4.clbpnorm(i64 %a)
  ret i32 %z
}
; CHECK: r0 = normamt(r1:0)

declare i32 @llvm.hexagon.S4.clbpaddi(i64, i32)
define i32 @S4_clbpaddi(i64 %a) {
  %z = call i32 @llvm.hexagon.S4.clbpaddi(i64 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = add(clb(r1:0), #0)

declare i32 @llvm.hexagon.S4.clbaddi(i32, i32)
define i32 @S4_clbaddi(i32 %a) {
  %z = call i32 @llvm.hexagon.S4.clbaddi(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = add(clb(r0), #0)

declare i32 @llvm.hexagon.S2.cl0(i32)
define i32 @S2_cl0(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.cl0(i32 %a)
  ret i32 %z
}
; CHECK: r0 = cl0(r0)

declare i32 @llvm.hexagon.S2.cl1(i32)
define i32 @S2_cl1(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.cl1(i32 %a)
  ret i32 %z
}
; CHECK: r0 = cl1(r0)

declare i32 @llvm.hexagon.S2.clbnorm(i32)
define i32 @S4_clbnorm(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.clbnorm(i32 %a)
  ret i32 %z
}
; CHECK: r0 = normamt(r0)

; Count population
declare i32 @llvm.hexagon.S5.popcountp(i64)
define i32 @S5_popcountp(i64 %a) {
  %z = call i32 @llvm.hexagon.S5.popcountp(i64 %a)
  ret i32 %z
}
; CHECK: r0 = popcount(r1:0)

; Count trailing
declare i32 @llvm.hexagon.S2.ct0p(i64)
define i32 @S2_ct0p(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.ct0p(i64 %a)
  ret i32 %z
}
; CHECK: r0 = ct0(r1:0)

declare i32 @llvm.hexagon.S2.ct1p(i64)
define i32 @S2_ct1p(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.ct1p(i64 %a)
  ret i32 %z
}
; CHECK: r0 = ct1(r1:0)

declare i32 @llvm.hexagon.S2.ct0(i32)
define i32 @S2_ct0(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.ct0(i32 %a)
  ret i32 %z
}
; CHECK: r0 = ct0(r0)

declare i32 @llvm.hexagon.S2.ct1(i32)
define i32 @S2_ct1(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.ct1(i32 %a)
  ret i32 %z
}
; CHECK: r0 = ct1(r0)

; Extract bitfield
declare i64 @llvm.hexagon.S2.extractup(i64, i32, i32)
define i64 @S2_extractup(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.extractup(i64 %a, i32 0, i32 0)
  ret i64 %z
}
; CHECK: r1:0 = extractu(r1:0, #0, #0)

declare i64 @llvm.hexagon.S4.extractp(i64, i32, i32)
define i64 @S2_extractp(i64 %a) {
  %z = call i64 @llvm.hexagon.S4.extractp(i64 %a, i32 0, i32 0)
  ret i64 %z
}
; CHECK: r1:0 = extract(r1:0, #0, #0)

declare i32 @llvm.hexagon.S2.extractu(i32, i32, i32)
define i32 @S2_extractu(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.extractu(i32 %a, i32 0, i32 0)
  ret i32 %z
}
; CHECK: r0 = extractu(r0, #0, #0)

declare i32 @llvm.hexagon.S4.extract(i32, i32, i32)
define i32 @S2_extract(i32 %a) {
  %z = call i32 @llvm.hexagon.S4.extract(i32 %a, i32 0, i32 0)
  ret i32 %z
}
; CHECK: r0 = extract(r0, #0, #0)

declare i64 @llvm.hexagon.S2.extractup.rp(i64, i64)
define i64 @S2_extractup_rp(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.extractup.rp(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: r1:0 = extractu(r1:0, r3:2)

declare i64 @llvm.hexagon.S4.extractp.rp(i64, i64)
define i64 @S4_extractp_rp(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S4.extractp.rp(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: r1:0 = extract(r1:0, r3:2)

declare i32 @llvm.hexagon.S2.extractu.rp(i32, i64)
define i32 @S2_extractu_rp(i32 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.S2.extractu.rp(i32 %a, i64 %b)
  ret i32 %z
}
; CHECK: r0 = extractu(r0, r3:2)

declare i32 @llvm.hexagon.S4.extract.rp(i32, i64)
define i32 @S4_extract_rp(i32 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.S4.extract.rp(i32 %a, i64 %b)
  ret i32 %z
}
; CHECK: r0 = extract(r0, r3:2)

; Insert bitfield
declare i64 @llvm.hexagon.S2.insertp(i64, i64, i32, i32)
define i64 @S2_insertp(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.insertp(i64 %a, i64 %b, i32 0, i32 0)
  ret i64 %z
}
; CHECK: r1:0 = insert(r3:2, #0, #0)

declare i32 @llvm.hexagon.S2.insert(i32, i32, i32, i32)
define i32 @S2_insert(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.insert(i32 %a, i32 %b, i32 0, i32 0)
  ret i32 %z
}
; CHECK: r0 = insert(r1, #0, #0)

declare i32 @llvm.hexagon.S2.insert.rp(i32, i32, i64)
define i32 @S2_insert_rp(i32 %a, i32 %b, i64 %c) {
  %z = call i32 @llvm.hexagon.S2.insert.rp(i32 %a, i32 %b, i64 %c)
  ret i32 %z
}
; CHECK: r0 = insert(r1, r3:2)

declare i64 @llvm.hexagon.S2.insertp.rp(i64, i64, i64)
define i64 @S2_insertp_rp(i64 %a, i64 %b, i64 %c) {
  %z = call i64 @llvm.hexagon.S2.insertp.rp(i64 %a, i64 %b, i64 %c)
  ret i64 %z
}
; CHECK: r1:0 = insert(r3:2, r5:4)

; Interleave/deinterleave
declare i64 @llvm.hexagon.S2.deinterleave(i64)
define i64 @S2_deinterleave(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.deinterleave(i64 %a)
  ret i64 %z
}
; CHECK: r1:0 = deinterleave(r1:0)

declare i64 @llvm.hexagon.S2.interleave(i64)
define i64 @S2_interleave(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.interleave(i64 %a)
  ret i64 %z
}
; CHECK: r1:0 = interleave(r1:0)

; Linear feedback-shift operation
declare i64 @llvm.hexagon.S2.lfsp(i64, i64)
define i64 @S2_lfsp(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.lfsp(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: r1:0 = lfs(r1:0, r3:2)

; Masked parity
declare i32 @llvm.hexagon.S2.parityp(i64, i64)
define i32 @S2_parityp(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.S2.parityp(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: r0 = parity(r1:0, r3:2)

declare i32 @llvm.hexagon.S4.parity(i32, i32)
define i32 @S4_parity(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S4.parity(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = parity(r0, r1)

; Bit reverse
declare i64 @llvm.hexagon.S2.brevp(i64)
define i64 @S2_brevp(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.brevp(i64 %a)
  ret i64 %z
}
; CHECK: r1:0 = brev(r1:0)

declare i32 @llvm.hexagon.S2.brev(i32)
define i32 @S2_brev(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.brev(i32 %a)
  ret i32 %z
}
; CHECK: r0 = brev(r0)

; Set/clear/toggle bit
declare i32 @llvm.hexagon.S2.setbit.i(i32, i32)
define i32 @S2_setbit_i(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.setbit.i(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = setbit(r0, #0)

declare i32 @llvm.hexagon.S2.clrbit.i(i32, i32)
define i32 @S2_clrbit_i(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.clrbit.i(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = clrbit(r0, #0)

declare i32 @llvm.hexagon.S2.togglebit.i(i32, i32)
define i32 @S2_togglebit_i(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.togglebit.i(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = togglebit(r0, #0)

declare i32 @llvm.hexagon.S2.setbit.r(i32, i32)
define i32 @S2_setbit_r(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.setbit.r(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = setbit(r0, r1)

declare i32 @llvm.hexagon.S2.clrbit.r(i32, i32)
define i32 @S2_clrbit_r(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.clrbit.r(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = clrbit(r0, r1)

declare i32 @llvm.hexagon.S2.togglebit.r(i32, i32)
define i32 @S2_togglebit_r(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.togglebit.r(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = togglebit(r0, r1)

; Split bitfield
declare i64 @llvm.hexagon.A4.bitspliti(i32, i32)
define i64 @A4_bitspliti(i32 %a) {
  %z = call i64 @llvm.hexagon.A4.bitspliti(i32 %a, i32 0)
  ret i64 %z
}
; CHECK:  = bitsplit(r0, #0)

declare i64 @llvm.hexagon.A4.bitsplit(i32, i32)
define i64 @A4_bitsplit(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.A4.bitsplit(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = bitsplit(r0, r1)

; Table index
declare i32 @llvm.hexagon.S2.tableidxb.goodsyntax(i32, i32, i32, i32)
define i32 @S2_tableidxb_goodsyntax(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.tableidxb.goodsyntax(i32 %a, i32 %b, i32 0, i32 0)
  ret i32 %z
}
; CHECK: r0 = tableidxb(r1, #0, #0)

declare i32 @llvm.hexagon.S2.tableidxh.goodsyntax(i32, i32, i32, i32)
define i32 @S2_tableidxh_goodsyntax(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.tableidxh.goodsyntax(i32 %a, i32 %b, i32 0, i32 0)
  ret i32 %z
}
; CHECK: r0 = tableidxh(r1, #0, #-1)

declare i32 @llvm.hexagon.S2.tableidxw.goodsyntax(i32, i32, i32, i32)
define i32 @S2_tableidxw_goodsyntax(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.tableidxw.goodsyntax(i32 %a, i32 %b, i32 0, i32 0)
  ret i32 %z
}
; CHECK: r0 = tableidxw(r1, #0, #-2)

declare i32 @llvm.hexagon.S2.tableidxd.goodsyntax(i32, i32, i32, i32)
define i32 @S2_tableidxd_goodsyntax(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.tableidxd.goodsyntax(i32 %a, i32 %b, i32 0, i32 0)
  ret i32 %z
}
; CHECK: r0 = tableidxd(r1, #0, #-3)

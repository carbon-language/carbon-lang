; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; Hexagon Programmer's Reference Manual 11.10.7 XTYPE/PRED

; Compare byte
declare i32 @llvm.hexagon.A4.cmpbgt(i32, i32)
define i32 @A4_cmpbgt(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.cmpbgt(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = cmpb.gt(r0, r1)

declare i32 @llvm.hexagon.A4.cmpbeq(i32, i32)
define i32 @A4_cmpbeq(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.cmpbeq(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = cmpb.eq(r0, r1)

declare i32 @llvm.hexagon.A4.cmpbgtu(i32, i32)
define i32 @A4_cmpbgtu(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.cmpbgtu(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = cmpb.gtu(r0, r1)

declare i32 @llvm.hexagon.A4.cmpbgti(i32, i32)
define i32 @A4_cmpbgti(i32 %a) {
  %z = call i32 @llvm.hexagon.A4.cmpbgti(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: p0 = cmpb.gt(r0, #0)

declare i32 @llvm.hexagon.A4.cmpbeqi(i32, i32)
define i32 @A4_cmpbeqi(i32 %a) {
  %z = call i32 @llvm.hexagon.A4.cmpbeqi(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: p0 = cmpb.eq(r0, #0)

declare i32 @llvm.hexagon.A4.cmpbgtui(i32, i32)
define i32 @A4_cmpbgtui(i32 %a) {
  %z = call i32 @llvm.hexagon.A4.cmpbgtui(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: p0 = cmpb.gtu(r0, #0)

; Compare half
declare i32 @llvm.hexagon.A4.cmphgt(i32, i32)
define i32 @A4_cmphgt(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.cmphgt(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = cmph.gt(r0, r1)

declare i32 @llvm.hexagon.A4.cmpheq(i32, i32)
define i32 @A4_cmpheq(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.cmpheq(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = cmph.eq(r0, r1)

declare i32 @llvm.hexagon.A4.cmphgtu(i32, i32)
define i32 @A4_cmphgtu(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.cmphgtu(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = cmph.gtu(r0, r1)

declare i32 @llvm.hexagon.A4.cmphgti(i32, i32)
define i32 @A4_cmphgti(i32 %a) {
  %z = call i32 @llvm.hexagon.A4.cmphgti(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: p0 = cmph.gt(r0, #0)

declare i32 @llvm.hexagon.A4.cmpheqi(i32, i32)
define i32 @A4_cmpheqi(i32 %a) {
  %z = call i32 @llvm.hexagon.A4.cmpheqi(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: p0 = cmph.eq(r0, #0)

declare i32 @llvm.hexagon.A4.cmphgtui(i32, i32)
define i32 @A4_cmphgtui(i32 %a) {
  %z = call i32 @llvm.hexagon.A4.cmphgtui(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: p0 = cmph.gtu(r0, #0)

; Compare doublewords
declare i32 @llvm.hexagon.C2.cmpgtp(i64, i64)
define i32 @C2_cmpgtp(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.C2.cmpgtp(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: p0 = cmp.gt(r1:0, r3:2)

declare i32 @llvm.hexagon.C2.cmpeqp(i64, i64)
define i32 @C2_cmpeqp(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.C2.cmpeqp(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: p0 = cmp.eq(r1:0, r3:2)

declare i32 @llvm.hexagon.C2.cmpgtup(i64, i64)
define i32 @C2_cmpgtup(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.C2.cmpgtup(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: p0 = cmp.gtu(r1:0, r3:2)

; Compare bitmask
declare i32 @llvm.hexagon.C2.bitsclri(i32, i32)
define i32 @C2_bitsclri(i32 %a) {
  %z = call i32 @llvm.hexagon.C2.bitsclri(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: p0 = bitsclr(r0, #0)

declare i32 @llvm.hexagon.C4.nbitsclri(i32, i32)
define i32 @C4_nbitsclri(i32 %a) {
  %z = call i32 @llvm.hexagon.C4.nbitsclri(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: p0 = !bitsclr(r0, #0)

declare i32 @llvm.hexagon.C2.bitsset(i32, i32)
define i32 @C2_bitsset(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.C2.bitsset(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = bitsset(r0, r1)

declare i32 @llvm.hexagon.C4.nbitsset(i32, i32)
define i32 @C4_nbitsset(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.C4.nbitsset(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = !bitsset(r0, r1)

declare i32 @llvm.hexagon.C2.bitsclr(i32, i32)
define i32 @C2_bitsclr(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.C2.bitsclr(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = bitsclr(r0, r1)

declare i32 @llvm.hexagon.C4.nbitsclr(i32, i32)
define i32 @C4_nbitsclr(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.C4.nbitsclr(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = !bitsclr(r0, r1)

; Mask generate from predicate
declare i64 @llvm.hexagon.C2.mask(i32)
define i64 @C2_mask(i32 %a) {
  %z = call i64 @llvm.hexagon.C2.mask(i32 %a)
  ret i64 %z
}
; CHECK:  = mask(r0)

; Check for TLB match
declare i32 @llvm.hexagon.A4.tlbmatch(i64, i32)
define i32 @A4_tlbmatch(i64 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.tlbmatch(i64 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = tlbmatch(r1:0, r2)

; Test bit
declare i32 @llvm.hexagon.S2.tstbit.i(i32, i32)
define i32 @S2_tstbit_i(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.tstbit.i(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: p0 = tstbit(r0, #0)

declare i32 @llvm.hexagon.S4.ntstbit.i(i32, i32)
define i32 @S4_ntstbit_i(i32 %a) {
  %z = call i32 @llvm.hexagon.S4.ntstbit.i(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: p0 = !tstbit(r0, #0)

declare i32 @llvm.hexagon.S2.tstbit.r(i32, i32)
define i32 @S2_tstbit_r(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.tstbit.r(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = tstbit(r0, r1)

declare i32 @llvm.hexagon.S4.ntstbit.r(i32, i32)
define i32 @S4_ntstbit_r(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S4.ntstbit.r(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = !tstbit(r0, r1)

; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; Hexagon Programmer's Reference Manual 11.1.2 ALU32/PERM

; Combine words into doubleword
declare i64 @llvm.hexagon.A4.combineri(i32, i32)
define i64 @A4_combineri(i32 %a) {
  %z = call i64 @llvm.hexagon.A4.combineri(i32 %a, i32 0)
  ret i64 %z
}
; CHECK:  = combine(r0, #0)

declare i64 @llvm.hexagon.A4.combineir(i32, i32)
define i64 @A4_combineir(i32 %a) {
  %z = call i64 @llvm.hexagon.A4.combineir(i32 0, i32 %a)
  ret i64 %z
}
; CHECK:  = combine(#0, r0)

declare i64 @llvm.hexagon.A2.combineii(i32, i32)
define i64 @A2_combineii() {
  %z = call i64 @llvm.hexagon.A2.combineii(i32 0, i32 0)
  ret i64 %z
}
; CHECK: r1:0 = combine(#0, #0)

declare i32 @llvm.hexagon.A2.combine.hh(i32, i32)
define i32 @A2_combine_hh(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.combine.hh(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = combine(r0.h, r1.h)

declare i32 @llvm.hexagon.A2.combine.hl(i32, i32)
define i32 @A2_combine_hl(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.combine.hl(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = combine(r0.h, r1.l)

declare i32 @llvm.hexagon.A2.combine.lh(i32, i32)
define i32 @A2_combine_lh(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.combine.lh(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = combine(r0.l, r1.h)

declare i32 @llvm.hexagon.A2.combine.ll(i32, i32)
define i32 @A2_combine_ll(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.combine.ll(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = combine(r0.l, r1.l)

declare i64 @llvm.hexagon.A2.combinew(i32, i32)
define i64 @A2_combinew(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.A2.combinew(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = combine(r0, r1)

; Mux
declare i32 @llvm.hexagon.C2.muxri(i32, i32, i32)
define i32 @C2_muxri(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.C2.muxri(i32 %a, i32 0, i32 %b)
  ret i32 %z
}
; CHECK: r0 = mux(p0, #0, r1)

declare i32 @llvm.hexagon.C2.muxir(i32, i32, i32)
define i32 @C2_muxir(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.C2.muxir(i32 %a, i32 %b, i32 0)
  ret i32 %z
}
; CHECK: r0 = mux(p0, r1, #0)

declare i32 @llvm.hexagon.C2.mux(i32, i32, i32)
define i32 @C2_mux(i32 %a, i32 %b, i32 %c) {
  %z = call i32 @llvm.hexagon.C2.mux(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: r0 = mux(p0, r1, r2)

; Shift word by 16
declare i32 @llvm.hexagon.A2.aslh(i32)
define i32 @A2_aslh(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.aslh(i32 %a)
  ret i32 %z
}
; CHECK: r0 = aslh(r0)

declare i32 @llvm.hexagon.A2.asrh(i32)
define i32 @A2_asrh(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.asrh(i32 %a)
  ret i32 %z
}
; CHECK: r0 = asrh(r0)

; Pack high and low halfwords
declare i64 @llvm.hexagon.S2.packhl(i32, i32)
define i64 @S2_packhl(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.S2.packhl(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: r1:0 = packhl(r0, r1)

; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; Hexagon Programmer's Reference Manual 11.1.1 ALU32/ALU

; Add
declare i32 @llvm.hexagon.A2.addi(i32, i32)
define i32 @A2_addi(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.addi(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0 = add(r0, #0)

declare i32 @llvm.hexagon.A2.add(i32, i32)
define i32 @A2_add(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.add(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = add(r0, r1)

declare i32 @llvm.hexagon.A2.addsat(i32, i32)
define i32 @A2_addsat(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.addsat(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = add(r0, r1):sat

; Logical operations
declare i32 @llvm.hexagon.A2.and(i32, i32)
define i32 @A2_and(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.and(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = and(r0, r1)

declare i32 @llvm.hexagon.A2.or(i32, i32)
define i32 @A2_or(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.or(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = or(r0, r1)

declare i32 @llvm.hexagon.A2.xor(i32, i32)
define i32 @A2_xor(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.xor(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = xor(r0, r1)

declare i32 @llvm.hexagon.A4.andn(i32, i32)
define i32 @A4_andn(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.andn(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = and(r0, ~r1)

declare i32 @llvm.hexagon.A4.orn(i32, i32)
define i32 @A4_orn(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.orn(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = or(r0, ~r1)

; Nop
declare void @llvm.hexagon.A2.nop()
define void @A2_nop(i32 %a, i32 %b) {
  call void @llvm.hexagon.A2.nop()
  ret void
}
; CHECK: nop

; Subtract
declare i32 @llvm.hexagon.A2.sub(i32, i32)
define i32 @A2_sub(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.sub(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = sub(r0, r1)

declare i32 @llvm.hexagon.A2.subsat(i32, i32)
define i32 @A2_subsat(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.subsat(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: r0 = sub(r0, r1):sat

; Sign extend
declare i32 @llvm.hexagon.A2.sxtb(i32)
define i32 @A2_sxtb(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.sxtb(i32 %a)
  ret i32 %z
}
; CHECK: r0 = sxtb(r0)

declare i32 @llvm.hexagon.A2.sxth(i32)
define i32 @A2_sxth(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.sxth(i32 %a)
  ret i32 %z
}
; CHECK: r0 = sxth(r0)

; Transfer immediate
declare i32 @llvm.hexagon.A2.tfril(i32, i32)
define i32 @A2_tfril(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.tfril(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0.l = #0

declare i32 @llvm.hexagon.A2.tfrih(i32, i32)
define i32 @A2_tfrih(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.tfrih(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: r0.h = #0

declare i32 @llvm.hexagon.A2.tfrsi(i32)
define i32 @A2_tfrsi() {
  %z = call i32 @llvm.hexagon.A2.tfrsi(i32 0)
  ret i32 %z
}
; CHECK: r0 = #0

; Transfer register
declare i32 @llvm.hexagon.A2.tfr(i32)
define i32 @A2_tfr(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.tfr(i32 %a)
  ret i32 %z
}
; CHECK: r0 = r0

; Zero extend
declare i32 @llvm.hexagon.A2.zxth(i32)
define i32 @A2_zxth(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.zxth(i32 %a)
  ret i32 %z
}
; CHECK: r0 = zxth(r0)

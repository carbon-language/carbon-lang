; RUN: llc -march=hexagon < %s | FileCheck %s
; Hexagon Programmer's Reference Manual 11.2 CR

; Corner detection acceleration
declare i32 @llvm.hexagon.C4.fastcorner9(i32, i32)
define i32 @C4_fastcorner9(i32 %a, i32 %b) {
  %z = call i32@llvm.hexagon.C4.fastcorner9(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = fastcorner9(r0, r1)

declare i32 @llvm.hexagon.C4.fastcorner9.not(i32, i32)
define i32 @C4_fastcorner9_not(i32 %a, i32 %b) {
  %z = call i32@llvm.hexagon.C4.fastcorner9.not(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = !fastcorner9(r0, r1)

; Logical reductions on predicates
declare i32 @llvm.hexagon.C2.any8(i32)
define i32 @C2_any8(i32 %a) {
  %z = call i32@llvm.hexagon.C2.any8(i32 %a)
  ret i32 %z
}
; CHECK: p0 = any8(r0)

declare i32 @llvm.hexagon.C2.all8(i32)
define i32 @C2_all8(i32 %a) {
  %z = call i32@llvm.hexagon.C2.all8(i32 %a)
  ret i32 %z
}

; CHECK: p0 = all8(r0)

; Logical operations on predicates
declare i32 @llvm.hexagon.C2.and(i32, i32)
define i32 @C2_and(i32 %a, i32 %b) {
  %z = call i32@llvm.hexagon.C2.and(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = and(r0, r1)

declare i32 @llvm.hexagon.C2.or(i32, i32)
define i32 @C2_or(i32 %a, i32 %b) {
  %z = call i32@llvm.hexagon.C2.or(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = or(r0, r1)

declare i32 @llvm.hexagon.C2.xor(i32, i32)
define i32 @C2_xor(i32 %a, i32 %b) {
  %z = call i32@llvm.hexagon.C2.xor(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = xor(r0, r1)

declare i32 @llvm.hexagon.C2.andn(i32, i32)
define i32 @C2_andn(i32 %a, i32 %b) {
  %z = call i32@llvm.hexagon.C2.andn(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = and(r0, !r1)

declare i32 @llvm.hexagon.C2.not(i32)
define i32 @C2_not(i32 %a) {
  %z = call i32@llvm.hexagon.C2.not(i32 %a)
  ret i32 %z
}
; CHECK: p0 = not(r0)

declare i32 @llvm.hexagon.C2.orn(i32, i32)
define i32 @C2_orn(i32 %a, i32 %b) {
  %z = call i32@llvm.hexagon.C2.orn(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: p0 = or(r0, !r1)

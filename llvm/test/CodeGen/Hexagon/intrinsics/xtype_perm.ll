; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; Hexagon Programmer's Reference Manual 11.10.6 XTYPE/PERM

; Saturate
declare i32 @llvm.hexagon.A2.sat(i64)
define i32 @A2_sat(i64 %a) {
  %z = call i32 @llvm.hexagon.A2.sat(i64 %a)
  ret i32 %z
}
; CHECK: r0 = sat(r1:0)

declare i32 @llvm.hexagon.A2.sath(i32)
define i32 @A2_sath(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.sath(i32 %a)
  ret i32 %z
}
; CHECK: r0 = sath(r0)

declare i32 @llvm.hexagon.A2.satuh(i32)
define i32 @A2_satuh(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.satuh(i32 %a)
  ret i32 %z
}
; CHECK: r0 = satuh(r0)

declare i32 @llvm.hexagon.A2.satub(i32)
define i32 @A2_satub(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.satub(i32 %a)
  ret i32 %z
}
; CHECK: r0 = satub(r0)

declare i32 @llvm.hexagon.A2.satb(i32)
define i32 @A2_satb(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.satb(i32 %a)
  ret i32 %z
}
; CHECK: r0 = satb(r0)

; Swizzle bytes
declare i32 @llvm.hexagon.A2.swiz(i32)
define i32 @A2_swiz(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.swiz(i32 %a)
  ret i32 %z
}
; CHECK: r0 = swiz(r0)

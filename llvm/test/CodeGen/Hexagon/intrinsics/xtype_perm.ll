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

; Vector round and pack
declare i32 @llvm.hexagon.S2.vrndpackwh(i64)
define i32 @S2_vrndpackwh(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.vrndpackwh(i64 %a)
  ret i32 %z
}
; CHECK: r0 = vrndwh(r1:0)

declare i32 @llvm.hexagon.S2.vrndpackwhs(i64)
define i32 @S2_vrndpackwhs(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.vrndpackwhs(i64 %a)
  ret i32 %z
}
; CHECK: r0 = vrndwh(r1:0):sat

; Vector saturate and pack
declare i32 @llvm.hexagon.S2.vsathub(i64)
define i32 @S2_vsathub(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.vsathub(i64 %a)
  ret i32 %z
}
; CHECK: r0 = vsathub(r1:0)

declare i32 @llvm.hexagon.S2.vsatwh(i64)
define i32 @S2_vsatwh(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.vsatwh(i64 %a)
  ret i32 %z
}
; CHECK: r0 = vsatwh(r1:0)

declare i32 @llvm.hexagon.S2.vsatwuh(i64)
define i32 @S2_vsatwuh(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.vsatwuh(i64 %a)
  ret i32 %z
}
; CHECK: r0 = vsatwuh(r1:0)

declare i32 @llvm.hexagon.S2.vsathb(i64)
define i32 @S2_vsathb(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.vsathb(i64 %a)
  ret i32 %z
}
; CHECK: r0 = vsathb(r1:0)

declare i32 @llvm.hexagon.S2.svsathb(i32)
define i32 @S2_svsathb(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.svsathb(i32 %a)
  ret i32 %z
}
; CHECK: r0 = vsathb(r0)

declare i32 @llvm.hexagon.S2.svsathub(i32)
define i32 @S2_svsathub(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.svsathub(i32 %a)
  ret i32 %z
}
; CHECK: r0 = vsathub(r0)

; Vector saturate without pack
declare i64 @llvm.hexagon.S2.vsathub.nopack(i64)
define i64 @S2_vsathub_nopack(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.vsathub.nopack(i64 %a)
  ret i64 %z
}
; CHECK: r1:0 = vsathub(r1:0)

declare i64 @llvm.hexagon.S2.vsatwuh.nopack(i64)
define i64 @S2_vsatwuh_nopack(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.vsatwuh.nopack(i64 %a)
  ret i64 %z
}
; CHECK: r1:0 = vsatwuh(r1:0)

declare i64 @llvm.hexagon.S2.vsatwh.nopack(i64)
define i64 @S2_vsatwh_nopack(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.vsatwh.nopack(i64 %a)
  ret i64 %z
}
; CHECK: r1:0 = vsatwh(r1:0)

declare i64 @llvm.hexagon.S2.vsathb.nopack(i64)
define i64 @S2_vsathb_nopack(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.vsathb.nopack(i64 %a)
  ret i64 %z
}
; CHECK: r1:0 = vsathb(r1:0)

; Vector shuffle
declare i64 @llvm.hexagon.S2.shuffeb(i64, i64)
define i64 @S2_shuffeb(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.shuffeb(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: r1:0 = shuffeb(r1:0, r3:2)

declare i64 @llvm.hexagon.S2.shuffob(i64, i64)
define i64 @S2_shuffob(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.shuffob(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: r1:0 = shuffob(r1:0, r3:2)

declare i64 @llvm.hexagon.S2.shuffeh(i64, i64)
define i64 @S2_shuffeh(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.shuffeh(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: r1:0 = shuffeh(r1:0, r3:2)

declare i64 @llvm.hexagon.S2.shuffoh(i64, i64)
define i64 @S2_shuffoh(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.shuffoh(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: r1:0 = shuffoh(r1:0, r3:2)

; Vector splat bytes
declare i32 @llvm.hexagon.S2.vsplatrb(i32)
define i32 @S2_vsplatrb(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.vsplatrb(i32 %a)
  ret i32 %z
}
; CHECK: r0 = vsplatb(r0)

; Vector splat halfwords
declare i64 @llvm.hexagon.S2.vsplatrh(i32)
define i64 @S2_vsplatrh(i32 %a) {
  %z = call i64 @llvm.hexagon.S2.vsplatrh(i32 %a)
  ret i64 %z
}
; CHECK:  = vsplath(r0)

; Vector splice
declare i64 @llvm.hexagon.S2.vspliceib(i64, i64, i32)
define i64 @S2_vspliceib(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.vspliceib(i64 %a, i64 %b, i32 0)
  ret i64 %z
}
; CHECK: r1:0 = vspliceb(r1:0, r3:2, #0)

declare i64 @llvm.hexagon.S2.vsplicerb(i64, i64, i32)
define i64 @S2_vsplicerb(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.vsplicerb(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: r1:0 = vspliceb(r1:0, r3:2, p0)

; Vector sign extend
declare i64 @llvm.hexagon.S2.vsxtbh(i32)
define i64 @S2_vsxtbh(i32 %a) {
  %z = call i64 @llvm.hexagon.S2.vsxtbh(i32 %a)
  ret i64 %z
}
; CHECK:  = vsxtbh(r0)

declare i64 @llvm.hexagon.S2.vsxthw(i32)
define i64 @S2_vsxthw(i32 %a) {
  %z = call i64 @llvm.hexagon.S2.vsxthw(i32 %a)
  ret i64 %z
}
; CHECK:  = vsxthw(r0)

; Vector truncate
declare i32 @llvm.hexagon.S2.vtrunohb(i64)
define i32 @S2_vtrunohb(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.vtrunohb(i64 %a)
  ret i32 %z
}
; CHECK: r0 = vtrunohb(r1:0)

declare i32 @llvm.hexagon.S2.vtrunehb(i64)
define i32 @S2_vtrunehb(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.vtrunehb(i64 %a)
  ret i32 %z
}
; CHECK: r0 = vtrunehb(r1:0)

declare i64 @llvm.hexagon.S2.vtrunowh(i64, i64)
define i64 @S2_vtrunowh(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.vtrunowh(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: r1:0 = vtrunowh(r1:0, r3:2)

declare i64 @llvm.hexagon.S2.vtrunewh(i64, i64)
define i64 @S2_vtrunewh(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.vtrunewh(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: r1:0 = vtrunewh(r1:0, r3:2)

; Vector zero extend
declare i64 @llvm.hexagon.S2.vzxtbh(i32)
define i64 @S2_vzxtbh(i32 %a) {
  %z = call i64 @llvm.hexagon.S2.vzxtbh(i32 %a)
  ret i64 %z
}
; CHECK:  = vzxtbh(r0)

declare i64 @llvm.hexagon.S2.vzxthw(i32)
define i64 @S2_vzxthw(i32 %a) {
  %z = call i64 @llvm.hexagon.S2.vzxthw(i32 %a)
  ret i64 %z
}
; CHECK:  = vzxthw(r0)

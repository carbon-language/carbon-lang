; RUN: llc -march=hexagon -mcpu=hexagonv5 -O0 < %s | FileCheck %s
; Hexagon Programmer's Reference Manual 11.10.4 XTYPE/FP

; Floating point addition
declare float @llvm.hexagon.F2.sfadd(float, float)
define float @F2_sfadd(float %a, float %b) {
  %z = call float @llvm.hexagon.F2.sfadd(float %a, float %b)
  ret float %z
}
; CHECK: r0 = sfadd(r0, r1)

; Classify floating-point value
declare i32 @llvm.hexagon.F2.sfclass(float, i32)
define i32 @F2_sfclass(float %a) {
  %z = call i32 @llvm.hexagon.F2.sfclass(float %a, i32 0)
  ret i32 %z
}
; CHECK: p0 = sfclass(r0, #0)

declare i32 @llvm.hexagon.F2.dfclass(double, i32)
define i32 @F2_dfclass(double %a) {
  %z = call i32 @llvm.hexagon.F2.dfclass(double %a, i32 0)
  ret i32 %z
}
; CHECK: p0 = dfclass(r1:0, #0)

; Compare floating-point value
declare i32 @llvm.hexagon.F2.sfcmpge(float, float)
define i32 @F2_sfcmpge(float %a, float %b) {
  %z = call i32 @llvm.hexagon.F2.sfcmpge(float %a, float %b)
  ret i32 %z
}
; CHECK: p0 = sfcmp.ge(r0, r1)

declare i32 @llvm.hexagon.F2.sfcmpuo(float, float)
define i32 @F2_sfcmpuo(float %a, float %b) {
  %z = call i32 @llvm.hexagon.F2.sfcmpuo(float %a, float %b)
  ret i32 %z
}
; CHECK: p0 = sfcmp.uo(r0, r1)

declare i32 @llvm.hexagon.F2.sfcmpeq(float, float)
define i32 @F2_sfcmpeq(float %a, float %b) {
  %z = call i32 @llvm.hexagon.F2.sfcmpeq(float %a, float %b)
  ret i32 %z
}
; CHECK: p0 = sfcmp.eq(r0, r1)

declare i32 @llvm.hexagon.F2.sfcmpgt(float, float)
define i32 @F2_sfcmpgt(float %a, float %b) {
  %z = call i32 @llvm.hexagon.F2.sfcmpgt(float %a, float %b)
  ret i32 %z
}
; CHECK: p0 = sfcmp.gt(r0, r1)

declare i32 @llvm.hexagon.F2.dfcmpge(double, double)
define i32 @F2_dfcmpge(double %a, double %b) {
  %z = call i32 @llvm.hexagon.F2.dfcmpge(double %a, double %b)
  ret i32 %z
}
; CHECK: p0 = dfcmp.ge(r1:0, r3:2)

declare i32 @llvm.hexagon.F2.dfcmpuo(double, double)
define i32 @F2_dfcmpuo(double %a, double %b) {
  %z = call i32 @llvm.hexagon.F2.dfcmpuo(double %a, double %b)
  ret i32 %z
}
; CHECK: p0 = dfcmp.uo(r1:0, r3:2)

declare i32 @llvm.hexagon.F2.dfcmpeq(double, double)
define i32 @F2_dfcmpeq(double %a, double %b) {
  %z = call i32 @llvm.hexagon.F2.dfcmpeq(double %a, double %b)
  ret i32 %z
}
; CHECK: p0 = dfcmp.eq(r1:0, r3:2)

declare i32 @llvm.hexagon.F2.dfcmpgt(double, double)
define i32 @F2_dfcmpgt(double %a, double %b) {
  %z = call i32 @llvm.hexagon.F2.dfcmpgt(double %a, double %b)
  ret i32 %z
}
; CHECK: p0 = dfcmp.gt(r1:0, r3:2)

; Convert floating-point value to other format
declare double @llvm.hexagon.F2.conv.sf2df(float)
define double @F2_conv_sf2df(float %a) {
  %z = call double @llvm.hexagon.F2.conv.sf2df(float %a)
  ret double %z
}
; CHECK:  = convert_sf2df(r0)

declare float @llvm.hexagon.F2.conv.df2sf(double)
define float @F2_conv_df2sf(double %a) {
  %z = call float @llvm.hexagon.F2.conv.df2sf(double %a)
  ret float %z
}
; CHECK: r0 = convert_df2sf(r1:0)

; Convert integer to floating-point value
declare double @llvm.hexagon.F2.conv.ud2df(i64)
define double @F2_conv_ud2df(i64 %a) {
  %z = call double @llvm.hexagon.F2.conv.ud2df(i64 %a)
  ret double %z
}
; CHECK: r1:0 = convert_ud2df(r1:0)

declare double @llvm.hexagon.F2.conv.d2df(i64)
define double @F2_conv_d2df(i64 %a) {
  %z = call double @llvm.hexagon.F2.conv.d2df(i64 %a)
  ret double %z
}
; CHECK: r1:0 = convert_d2df(r1:0)

declare double @llvm.hexagon.F2.conv.uw2df(i32)
define double @F2_conv_uw2df(i32 %a) {
  %z = call double @llvm.hexagon.F2.conv.uw2df(i32 %a)
  ret double %z
}
; CHECK:  = convert_uw2df(r0)

declare double @llvm.hexagon.F2.conv.w2df(i32)
define double @F2_conv_w2df(i32 %a) {
  %z = call double @llvm.hexagon.F2.conv.w2df(i32 %a)
  ret double %z
}
; CHECK:  = convert_w2df(r0)

declare float @llvm.hexagon.F2.conv.ud2sf(i64)
define float @F2_conv_ud2sf(i64 %a) {
  %z = call float @llvm.hexagon.F2.conv.ud2sf(i64 %a)
  ret float %z
}
; CHECK: r0 = convert_ud2sf(r1:0)

declare float @llvm.hexagon.F2.conv.d2sf(i64)
define float @F2_conv_d2sf(i64 %a) {
  %z = call float @llvm.hexagon.F2.conv.d2sf(i64 %a)
  ret float %z
}
; CHECK: r0 = convert_d2sf(r1:0)

declare float @llvm.hexagon.F2.conv.uw2sf(i32)
define float @F2_conv_uw2sf(i32 %a) {
  %z = call float @llvm.hexagon.F2.conv.uw2sf(i32 %a)
  ret float %z
}
; CHECK: r0 = convert_uw2sf(r0)

declare float @llvm.hexagon.F2.conv.w2sf(i32)
define float @F2_conv_w2sf(i32 %a) {
  %z = call float @llvm.hexagon.F2.conv.w2sf(i32 %a)
  ret float %z
}
; CHECK: r0 = convert_w2sf(r0)

; Convert floating-point value to integer
declare i64 @llvm.hexagon.F2.conv.df2d(double)
define i64 @F2_conv_df2d(double %a) {
  %z = call i64 @llvm.hexagon.F2.conv.df2d(double %a)
  ret i64 %z
}
; CHECK: r1:0 = convert_df2d(r1:0)

declare i64 @llvm.hexagon.F2.conv.df2ud(double)
define i64 @F2_conv_df2ud(double %a) {
  %z = call i64 @llvm.hexagon.F2.conv.df2ud(double %a)
  ret i64 %z
}
; CHECK: r1:0 = convert_df2ud(r1:0)

declare i64 @llvm.hexagon.F2.conv.df2d.chop(double)
define i64 @F2_conv_df2d_chop(double %a) {
  %z = call i64 @llvm.hexagon.F2.conv.df2d.chop(double %a)
  ret i64 %z
}
; CHECK: r1:0 = convert_df2d(r1:0):chop

declare i64 @llvm.hexagon.F2.conv.df2ud.chop(double)
define i64 @F2_conv_df2ud_chop(double %a) {
  %z = call i64 @llvm.hexagon.F2.conv.df2ud.chop(double %a)
  ret i64 %z
}
; CHECK: r1:0 = convert_df2ud(r1:0):chop

declare i64 @llvm.hexagon.F2.conv.sf2ud(float)
define i64 @F2_conv_sf2ud(float %a) {
  %z = call i64 @llvm.hexagon.F2.conv.sf2ud(float %a)
  ret i64 %z
}
; CHECK:  = convert_sf2ud(r0)

declare i64 @llvm.hexagon.F2.conv.sf2d(float)
define i64 @F2_conv_sf2d(float %a) {
  %z = call i64 @llvm.hexagon.F2.conv.sf2d(float %a)
  ret i64 %z
}
; CHECK:  = convert_sf2d(r0)

declare i64 @llvm.hexagon.F2.conv.sf2d.chop(float)
define i64 @F2_conv_sf2d_chop(float %a) {
  %z = call i64 @llvm.hexagon.F2.conv.sf2d.chop(float %a)
  ret i64 %z
}
; CHECK:  = convert_sf2d(r0):chop

declare i64 @llvm.hexagon.F2.conv.sf2ud.chop(float)
define i64 @F2_conv_sf2ud_chop(float %a) {
  %z = call i64 @llvm.hexagon.F2.conv.sf2ud.chop(float %a)
  ret i64 %z
}
; CHECK:  = convert_sf2ud(r0):chop

declare i32 @llvm.hexagon.F2.conv.df2uw(double)
define i32 @F2_conv_df2uw(double %a) {
  %z = call i32 @llvm.hexagon.F2.conv.df2uw(double %a)
  ret i32 %z
}
; CHECK: r0 = convert_df2uw(r1:0)

declare i32 @llvm.hexagon.F2.conv.df2w(double)
define i32 @F2_conv_df2w(double %a) {
  %z = call i32 @llvm.hexagon.F2.conv.df2w(double %a)
  ret i32 %z
}
; CHECK: r0 = convert_df2w(r1:0)

declare i32 @llvm.hexagon.F2.conv.df2w.chop(double)
define i32 @F2_conv_df2w_chop(double %a) {
  %z = call i32 @llvm.hexagon.F2.conv.df2w.chop(double %a)
  ret i32 %z
}
; CHECK: r0 = convert_df2w(r1:0):chop

declare i32 @llvm.hexagon.F2.conv.df2uw.chop(double)
define i32 @F2_conv_df2uw_chop(double %a) {
  %z = call i32 @llvm.hexagon.F2.conv.df2uw.chop(double %a)
  ret i32 %z
}
; CHECK: r0 = convert_df2uw(r1:0):chop

declare i32 @llvm.hexagon.F2.conv.sf2uw(float)
define i32 @F2_conv_sf2uw(float %a) {
  %z = call i32 @llvm.hexagon.F2.conv.sf2uw(float %a)
  ret i32 %z
}
; CHECK: r0 = convert_sf2uw(r0)

declare i32 @llvm.hexagon.F2.conv.sf2uw.chop(float)
define i32 @F2_conv_sf2uw_chop(float %a) {
  %z = call i32 @llvm.hexagon.F2.conv.sf2uw.chop(float %a)
  ret i32 %z
}
; CHECK: r0 = convert_sf2uw(r0):chop

declare i32 @llvm.hexagon.F2.conv.sf2w(float)
define i32 @F2_conv_sf2w(float %a) {
  %z = call i32 @llvm.hexagon.F2.conv.sf2w(float %a)
  ret i32 %z
}
; CHECK: r0 = convert_sf2w(r0)

declare i32 @llvm.hexagon.F2.conv.sf2w.chop(float)
define i32 @F2_conv_sf2w_chop(float %a) {
  %z = call i32 @llvm.hexagon.F2.conv.sf2w.chop(float %a)
  ret i32 %z
}
; CHECK: r0 = convert_sf2w(r0):chop

; Floating point extreme value assistance
declare float @llvm.hexagon.F2.sffixupr(float)
define float @F2_sffixupr(float %a) {
  %z = call float @llvm.hexagon.F2.sffixupr(float %a)
  ret float %z
}
; CHECK: r0 = sffixupr(r0)

declare float @llvm.hexagon.F2.sffixupn(float, float)
define float @F2_sffixupn(float %a, float %b) {
  %z = call float @llvm.hexagon.F2.sffixupn(float %a, float %b)
  ret float %z
}
; CHECK: r0 = sffixupn(r0, r1)

declare float @llvm.hexagon.F2.sffixupd(float, float)
define float @F2_sffixupd(float %a, float %b) {
  %z = call float @llvm.hexagon.F2.sffixupd(float %a, float %b)
  ret float %z
}
; CHECK: r0 = sffixupd(r0, r1)

; Floating point fused multiply-add
declare float @llvm.hexagon.F2.sffma(float, float, float)
define float @F2_sffma(float %a, float %b, float %c) {
  %z = call float @llvm.hexagon.F2.sffma(float %a, float %b, float %c)
  ret float %z
}
; CHECK: r0 += sfmpy(r1, r2)

declare float @llvm.hexagon.F2.sffms(float, float, float)
define float @F2_sffms(float %a, float %b, float %c) {
  %z = call float @llvm.hexagon.F2.sffms(float %a, float %b, float %c)
  ret float %z
}
; CHECK: r0 -= sfmpy(r1, r2)

; Floating point fused multiply-add with scaling
declare float @llvm.hexagon.F2.sffma.sc(float, float, float, i32)
define float @F2_sffma_sc(float %a, float %b, float %c, i32 %d) {
  %z = call float @llvm.hexagon.F2.sffma.sc(float %a, float %b, float %c, i32 %d)
  ret float %z
}
; CHECK: r0 += sfmpy(r1, r2, p0):scale

; Floating point fused multiply-add for library routines
declare float @llvm.hexagon.F2.sffma.lib(float, float, float)
define float @F2_sffma_lib(float %a, float %b, float %c) {
  %z = call float @llvm.hexagon.F2.sffma.lib(float %a, float %b, float %c)
  ret float %z
}
; CHECK: r0 += sfmpy(r1, r2):lib

declare float @llvm.hexagon.F2.sffms.lib(float, float, float)
define float @F2_sffms_lib(float %a, float %b, float %c) {
  %z = call float @llvm.hexagon.F2.sffms.lib(float %a, float %b, float %c)
  ret float %z
}
; CHECK: r0 -= sfmpy(r1, r2):lib

; Create floating-point constant
declare float @llvm.hexagon.F2.sfimm.p(i32)
define float @F2_sfimm_p() {
  %z = call float @llvm.hexagon.F2.sfimm.p(i32 0)
  ret float %z
}
; CHECK: r0 = sfmake(#0):pos

declare float @llvm.hexagon.F2.sfimm.n(i32)
define float @F2_sfimm_n() {
  %z = call float @llvm.hexagon.F2.sfimm.n(i32 0)
  ret float %z
} 
; CHECK: r0 = sfmake(#0):neg

declare double @llvm.hexagon.F2.dfimm.p(i32)
define double @F2_dfimm_p() {
  %z = call double @llvm.hexagon.F2.dfimm.p(i32 0)
  ret double %z
}
; CHECK: r1:0 = dfmake(#0):pos

declare double @llvm.hexagon.F2.dfimm.n(i32)
define double @F2_dfimm_n() {
  %z = call double @llvm.hexagon.F2.dfimm.n(i32 0)
  ret double %z
}
; CHECK: r1:0 = dfmake(#0):neg

; Floating point maximum
declare float @llvm.hexagon.F2.sfmax(float, float)
define float @F2_sfmax(float %a, float %b) {
  %z = call float @llvm.hexagon.F2.sfmax(float %a, float %b)
  ret float %z
}
; CHECK: r0 = sfmax(r0, r1)

; Floating point minimum
declare float @llvm.hexagon.F2.sfmin(float, float)
define float @F2_sfmin(float %a, float %b) {
  %z = call float @llvm.hexagon.F2.sfmin(float %a, float %b)
  ret float %z
}
; CHECK: r0 = sfmin(r0, r1)

; Floating point multiply
declare float @llvm.hexagon.F2.sfmpy(float, float)
define float @F2_sfmpy(float %a, float %b) {
  %z = call float @llvm.hexagon.F2.sfmpy(float %a, float %b)
  ret float %z
}
; CHECK: r0 = sfmpy(r0, r1)

; Floating point subtraction
declare float @llvm.hexagon.F2.sfsub(float, float)
define float @F2_sfsub(float %a, float %b) {
  %z = call float @llvm.hexagon.F2.sfsub(float %a, float %b)
  ret float %z
}
; CHECK: r0 = sfsub(r0, r1)

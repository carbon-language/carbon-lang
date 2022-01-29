; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 -asm-verbose=0 < %s | FileCheck %s

;
; FLOGB
;

define <vscale x 8 x i16> @flogb_f16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg, <vscale x 8 x half> %b) {
; CHECK-LABEL: flogb_f16:
; CHECK: flogb z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.flogb.nxv8f16(<vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @flogb_f32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, <vscale x 4 x float> %b) {
; CHECK-LABEL: flogb_f32:
; CHECK: flogb z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.flogb.nxv4f32(<vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x float> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @flogb_f64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, <vscale x 2 x double> %b) {
; CHECK-LABEL: flogb_f64:
; CHECK: flogb z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.flogb.nxv2f64(<vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x double> %b)
  ret <vscale x 2 x i64> %out
}

declare <vscale x 8 x i16> @llvm.aarch64.sve.flogb.nxv8f16(<vscale x 8 x i16>, <vscale x 8 x i1>, <vscale x 8 x half>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.flogb.nxv4f32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.flogb.nxv2f64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x double>)

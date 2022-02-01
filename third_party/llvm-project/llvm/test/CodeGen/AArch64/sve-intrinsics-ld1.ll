; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s
; RUN: llc -O0 -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; LD1B
;

define <vscale x 16 x i8> @ld1b_i8(<vscale x 16 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld1b_i8:
; CHECK: ld1b { z0.b }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1> %pred, i8* %addr)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @ld1b_h(<vscale x 8 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld1b_h:
; CHECK: ld1b { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 8 x i8> @llvm.aarch64.sve.ld1.nxv8i8(<vscale x 8 x i1> %pred, i8* %addr)
  %res = zext <vscale x 8 x i8> %load to <vscale x 8 x i16>
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @ld1sb_h(<vscale x 8 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld1sb_h:
; CHECK: ld1sb { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 8 x i8> @llvm.aarch64.sve.ld1.nxv8i8(<vscale x 8 x i1> %pred, i8* %addr)
  %res = sext <vscale x 8 x i8> %load to <vscale x 8 x i16>
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @ld1b_s(<vscale x 4 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld1b_s:
; CHECK: ld1b { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ld1.nxv4i8(<vscale x 4 x i1> %pred, i8* %addr)
  %res = zext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @ld1sb_s(<vscale x 4 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld1sb_s:
; CHECK: ld1sb { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ld1.nxv4i8(<vscale x 4 x i1> %pred, i8* %addr)
  %res = sext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @ld1b_d(<vscale x 2 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld1b_d:
; CHECK: ld1b { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.nxv2i8(<vscale x 2 x i1> %pred, i8* %addr)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @ld1sb_d(<vscale x 2 x i1> %pred, i8* %addr) {
; CHECK-LABEL: ld1sb_d:
; CHECK: ld1sb { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.nxv2i8(<vscale x 2 x i1> %pred, i8* %addr)
  %res = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

;
; LD1H
;

define <vscale x 8 x i16> @ld1h_i16(<vscale x 8 x i1> %pred, i16* %addr) {
; CHECK-LABEL: ld1h_i16:
; CHECK: ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.ld1.nxv8i16(<vscale x 8 x i1> %pred, i16* %addr)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x half> @ld1h_f16(<vscale x 8 x i1> %pred, half* %addr) {
; CHECK-LABEL: ld1h_f16:
; CHECK: ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x half> @llvm.aarch64.sve.ld1.nxv8f16(<vscale x 8 x i1> %pred, half* %addr)
  ret <vscale x 8 x half> %res
}

define <vscale x 8 x bfloat> @ld1h_bf16(<vscale x 8 x i1> %pred, bfloat* %addr) #0 {
; CHECK-LABEL: ld1h_bf16:
; CHECK: ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 8 x bfloat> @llvm.aarch64.sve.ld1.nxv8bf16(<vscale x 8 x i1> %pred, bfloat* %addr)
  ret <vscale x 8 x bfloat> %res
}

define <vscale x 4 x i32> @ld1h_s(<vscale x 4 x i1> %pred, i16* %addr) {
; CHECK-LABEL: ld1h_s:
; CHECK: ld1h { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.nxv4i16(<vscale x 4 x i1> %pred, i16* %addr)
  %res = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @ld1sh_s(<vscale x 4 x i1> %pred, i16* %addr) {
; CHECK-LABEL: ld1sh_s:
; CHECK: ld1sh { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.nxv4i16(<vscale x 4 x i1> %pred, i16* %addr)
  %res = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @ld1h_d(<vscale x 2 x i1> %pred, i16* %addr) {
; CHECK-LABEL: ld1h_d:
; CHECK: ld1h { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.nxv2i16(<vscale x 2 x i1> %pred, i16* %addr)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @ld1sh_d(<vscale x 2 x i1> %pred, i16* %addr) {
; CHECK-LABEL: ld1sh_d:
; CHECK: ld1sh { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.nxv2i16(<vscale x 2 x i1> %pred, i16* %addr)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

;
; LD1W
;

define <vscale x 4 x i32> @ld1w_i32(<vscale x 4 x i1> %pred, i32* %addr) {
; CHECK-LABEL: ld1w_i32:
; CHECK: ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1.nxv4i32(<vscale x 4 x i1> %pred, i32* %addr)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x float> @ld1w_f32(<vscale x 4 x i1> %pred, float* %addr) {
; CHECK-LABEL: ld1w_f32:
; CHECK: ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 4 x float> @llvm.aarch64.sve.ld1.nxv4f32(<vscale x 4 x i1> %pred, float* %addr)
  ret <vscale x 4 x float> %res
}

define <vscale x 2 x i64> @ld1w_d(<vscale x 2 x i1> %pred, i32* %addr) {
; CHECK-LABEL: ld1w_d:
; CHECK: ld1w { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.nxv2i32(<vscale x 2 x i1> %pred, i32* %addr)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @ld1sw_d(<vscale x 2 x i1> %pred, i32* %addr) {
; CHECK-LABEL: ld1sw_d:
; CHECK: ld1sw { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.nxv2i32(<vscale x 2 x i1> %pred, i32* %addr)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

;
; LD1D
;

define <vscale x 2 x i64> @ld1d_i64(<vscale x 2 x i1> %pred, i64* %addr) {
; CHECK-LABEL: ld1d_i64:
; CHECK: ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1.nxv2i64(<vscale x 2 x i1> %pred,
                                                               i64* %addr)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x double> @ld1d_f64(<vscale x 2 x i1> %pred, double* %addr) {
; CHECK-LABEL: ld1d_f64:
; CHECK: ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %res = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.nxv2f64(<vscale x 2 x i1> %pred,
                                                                  double* %addr)
  ret <vscale x 2 x double> %res
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1>, i8*)

declare <vscale x 8 x i8> @llvm.aarch64.sve.ld1.nxv8i8(<vscale x 8 x i1>, i8*)
declare <vscale x 8 x i16> @llvm.aarch64.sve.ld1.nxv8i16(<vscale x 8 x i1>, i16*)
declare <vscale x 8 x half> @llvm.aarch64.sve.ld1.nxv8f16(<vscale x 8 x i1>, half*)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.ld1.nxv8bf16(<vscale x 8 x i1>, bfloat*)

declare <vscale x 4 x i8> @llvm.aarch64.sve.ld1.nxv4i8(<vscale x 4 x i1>, i8*)
declare <vscale x 4 x i16> @llvm.aarch64.sve.ld1.nxv4i16(<vscale x 4 x i1>, i16*)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ld1.nxv4i32(<vscale x 4 x i1>, i32*)
declare <vscale x 4 x float> @llvm.aarch64.sve.ld1.nxv4f32(<vscale x 4 x i1>, float*)

declare <vscale x 2 x i8> @llvm.aarch64.sve.ld1.nxv2i8(<vscale x 2 x i1>, i8*)
declare <vscale x 2 x i16> @llvm.aarch64.sve.ld1.nxv2i16(<vscale x 2 x i1>, i16*)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ld1.nxv2i32(<vscale x 2 x i1>, i32*)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ld1.nxv2i64(<vscale x 2 x i1>, i64*)
declare <vscale x 2 x double> @llvm.aarch64.sve.ld1.nxv2f64(<vscale x 2 x i1>, double*)

; +bf16 is required for the bfloat version.
attributes #0 = { "target-features"="+sve,+bf16" }

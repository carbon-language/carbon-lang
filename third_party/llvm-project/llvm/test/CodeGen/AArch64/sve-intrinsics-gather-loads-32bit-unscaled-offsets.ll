; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; LD1B, LD1W, LD1H, LD1D: base + 32-bit unscaled offset, sign (sxtw) or zero
; (uxtw) extended to 64 bits.
;   e.g. ld1h { z0.d }, p0/z, [x0, z0.d, uxtw]
;

; LD1B
define <vscale x 4 x i32> @gld1b_s_uxtw(<vscale x 4 x i1> %pg, i8* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1b_s_uxtw:
; CHECK: ld1b { z0.s }, p0/z, [x0, z0.s, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ld1.gather.uxtw.nxv4i8(<vscale x 4 x i1> %pg,
                                                                          i8* %base,
                                                                          <vscale x 4 x i32> %b)
  %res = zext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @gld1b_s_sxtw(<vscale x 4 x i1> %pg, i8* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1b_s_sxtw:
; CHECK: ld1b { z0.s }, p0/z, [x0, z0.s, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ld1.gather.sxtw.nxv4i8(<vscale x 4 x i1> %pg,
                                                                          i8* %base,
                                                                          <vscale x 4 x i32> %b)
  %res = zext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gld1b_d_uxtw(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1b_d_uxtw:
; CHECK: ld1b { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.uxtw.nxv2i8(<vscale x 2 x i1> %pg,
                                                                          i8* %base,
                                                                          <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1b_d_sxtw(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1b_d_sxtw:
; CHECK: ld1b { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.sxtw.nxv2i8(<vscale x 2 x i1> %pg,
                                                                          i8* %base,
                                                                          <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LD1H
define <vscale x 4 x i32> @gld1h_s_uxtw(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1h_s_uxtw:
; CHECK: ld1h { z0.s }, p0/z, [x0, z0.s, uxtw]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.uxtw.nxv4i16(<vscale x 4 x i1> %pg,
                                                                            i16* %base,
                                                                            <vscale x 4 x i32> %b)
  %res = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @gld1h_s_sxtw(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1h_s_sxtw:
; CHECK: ld1h { z0.s }, p0/z, [x0, z0.s, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.sxtw.nxv4i16(<vscale x 4 x i1> %pg,
                                                                            i16* %base,
                                                                            <vscale x 4 x i32> %b)
  %res = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gld1h_d_uxtw(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1h_d_uxtw:
; CHECK: ld1h { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.uxtw.nxv2i16(<vscale x 2 x i1> %pg,
                                                                            i16* %base,
                                                                            <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1h_d_sxtw(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1h_d_sxtw:
; CHECK: ld1h { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.sxtw.nxv2i16(<vscale x 2 x i1> %pg,
                                                                            i16* %base,
                                                                            <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LD1W
define <vscale x 4 x i32> @gld1w_s_uxtw(<vscale x 4 x i1> %pg, i32* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1w_s_uxtw:
; CHECK: ld1w { z0.s }, p0/z, [x0, z0.s, uxtw]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.uxtw.nxv4i32(<vscale x 4 x i1> %pg,
                                                                            i32* %base,
                                                                            <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %load
}

define <vscale x 4 x i32> @gld1w_s_sxtw(<vscale x 4 x i1> %pg, i32* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1w_s_sxtw:
; CHECK: ld1w { z0.s }, p0/z, [x0, z0.s, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.nxv4i32(<vscale x 4 x i1> %pg,
                                                                            i32* %base,
                                                                            <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %load
}

define <vscale x 2 x i64> @gld1w_d_uxtw(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1w_d_uxtw:
; CHECK: ld1w { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.uxtw.nxv2i32(<vscale x 2 x i1> %pg,
                                                                            i32* %base,
                                                                            <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1w_d_sxtw(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1w_d_sxtw:
; CHECK: ld1w { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.nxv2i32(<vscale x 2 x i1> %pg,
                                                                            i32* %base,
                                                                            <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x float> @gld1w_s_uxtw_float(<vscale x 4 x i1> %pg, float* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1w_s_uxtw_float:
; CHECK: ld1w { z0.s }, p0/z, [x0, z0.s, uxtw]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x float> @llvm.aarch64.sve.ld1.gather.uxtw.nxv4f32(<vscale x 4 x i1> %pg,
                                                                              float* %base,
                                                                              <vscale x 4 x i32> %b)
  ret <vscale x 4 x float> %load
}

define <vscale x 4 x float> @gld1w_s_sxtw_float(<vscale x 4 x i1> %pg, float* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1w_s_sxtw_float:
; CHECK: ld1w { z0.s }, p0/z, [x0, z0.s, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x float> @llvm.aarch64.sve.ld1.gather.sxtw.nxv4f32(<vscale x 4 x i1> %pg,
                                                                              float* %base,
                                                                              <vscale x 4 x i32> %b)
  ret <vscale x 4 x float> %load
}

; LD1D
define <vscale x 2 x i64> @gld1d_d_uxtw(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1d_d_uxtw:
; CHECK: ld1d { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.uxtw.nxv2i64(<vscale x 2 x i1> %pg,
                                                                            i64* %base,
                                                                            <vscale x 2 x i32> %b)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x i64> @gld1d_d_sxtw(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1d_d_sxtw:
; CHECK: ld1d { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.sxtw.nxv2i64(<vscale x 2 x i1> %pg,
                                                                            i64* %base,
                                                                            <vscale x 2 x i32> %b)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @gld1d_d_uxtw_double(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1d_d_uxtw_double:
; CHECK: ld1d { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.uxtw.nxv2f64(<vscale x 2 x i1> %pg,
                                                                               double* %base,
                                                                               <vscale x 2 x i32> %b)
  ret <vscale x 2 x double> %load
}

define <vscale x 2 x double> @gld1d_d_sxtw_double(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1d_d_sxtw_double:
; CHECK: ld1d { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.sxtw.nxv2f64(<vscale x 2 x i1> %pg,
                                                                               double* %base,
                                                                               <vscale x 2 x i32> %b)
  ret <vscale x 2 x double> %load
}

;
; LD1SB, LD1SW, LD1SH: base + 32-bit unscaled offset, sign (sxtw) or zero
; (uxtw) extended to 64 bits.
;   e.g. ld1sh { z0.d }, p0/z, [x0, z0.d, uxtw]
;

; LD1SB
define <vscale x 4 x i32> @gld1sb_s_uxtw(<vscale x 4 x i1> %pg, i8* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1sb_s_uxtw:
; CHECK: ld1sb { z0.s }, p0/z, [x0, z0.s, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ld1.gather.uxtw.nxv4i8(<vscale x 4 x i1> %pg,
                                                                          i8* %base,
                                                                          <vscale x 4 x i32> %b)
  %res = sext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @gld1sb_s_sxtw(<vscale x 4 x i1> %pg, i8* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1sb_s_sxtw:
; CHECK: ld1sb { z0.s }, p0/z, [x0, z0.s, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ld1.gather.sxtw.nxv4i8(<vscale x 4 x i1> %pg,
                                                                          i8* %base,
                                                                          <vscale x 4 x i32> %b)
  %res = sext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gld1sb_d_uxtw(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1sb_d_uxtw:
; CHECK: ld1sb { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.uxtw.nxv2i8(<vscale x 2 x i1> %pg,
                                                                          i8* %base,
                                                                          <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1sb_d_sxtw(<vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1sb_d_sxtw:
; CHECK: ld1sb { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.sxtw.nxv2i8(<vscale x 2 x i1> %pg,
                                                                          i8* %base,
                                                                          <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LD1SH
define <vscale x 4 x i32> @gld1sh_s_uxtw(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1sh_s_uxtw:
; CHECK: ld1sh { z0.s }, p0/z, [x0, z0.s, uxtw]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.uxtw.nxv4i16(<vscale x 4 x i1> %pg,
                                                                            i16* %base,
                                                                            <vscale x 4 x i32> %b)
  %res = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @gld1sh_s_sxtw(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gld1sh_s_sxtw:
; CHECK: ld1sh { z0.s }, p0/z, [x0, z0.s, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.sxtw.nxv4i16(<vscale x 4 x i1> %pg,
                                                                            i16* %base,
                                                                            <vscale x 4 x i32> %b)
  %res = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gld1sh_d_uxtw(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1sh_d_uxtw:
; CHECK: ld1sh { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.uxtw.nxv2i16(<vscale x 2 x i1> %pg,
                                                                            i16* %base,
                                                                            <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1sh_d_sxtw(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1sh_d_sxtw:
; CHECK: ld1sh { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.sxtw.nxv2i16(<vscale x 2 x i1> %pg,
                                                                            i16* %base,
                                                                            <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LD1SW
define <vscale x 2 x i64> @gld1sw_d_uxtw(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1sw_d_uxtw:
; CHECK: ld1sw { z0.d }, p0/z, [x0, z0.d, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.uxtw.nxv2i32(<vscale x 2 x i1> %pg,
                                                                            i32* %base,
                                                                            <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1sw_d_sxtw(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gld1sw_d_sxtw:
; CHECK: ld1sw { z0.d }, p0/z, [x0, z0.d, sxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.nxv2i32(<vscale x 2 x i1> %pg,
                                                                            i32* %base,
                                                                            <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LD1B/LD1SB
declare <vscale x 4 x i8> @llvm.aarch64.sve.ld1.gather.uxtw.nxv4i8(<vscale x 4 x i1>, i8*, <vscale x 4 x i32>)
declare <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.uxtw.nxv2i8(<vscale x 2 x i1>, i8*, <vscale x 2 x i32>)
declare <vscale x 4 x i8> @llvm.aarch64.sve.ld1.gather.sxtw.nxv4i8(<vscale x 4 x i1>, i8*, <vscale x 4 x i32>)
declare <vscale x 2 x i8> @llvm.aarch64.sve.ld1.gather.sxtw.nxv2i8(<vscale x 2 x i1>, i8*, <vscale x 2 x i32>)

; LD1H/LD1SH
declare <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.sxtw.nxv4i16(<vscale x 4 x i1>, i16*, <vscale x 4 x i32>)
declare <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.sxtw.nxv2i16(<vscale x 2 x i1>, i16*, <vscale x 2 x i32>)
declare <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.uxtw.nxv4i16(<vscale x 4 x i1>, i16*, <vscale x 4 x i32>)
declare <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.uxtw.nxv2i16(<vscale x 2 x i1>, i16*, <vscale x 2 x i32>)

; LD1W/LD1SW
declare <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.nxv4i32(<vscale x 4 x i1>, i32*, <vscale x 4 x i32>)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.nxv2i32(<vscale x 2 x i1>, i32*, <vscale x 2 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.uxtw.nxv4i32(<vscale x 4 x i1>, i32*, <vscale x 4 x i32>)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.uxtw.nxv2i32(<vscale x 2 x i1>, i32*, <vscale x 2 x i32>)

declare <vscale x 4 x float> @llvm.aarch64.sve.ld1.gather.sxtw.nxv4f32(<vscale x 4 x i1>, float*, <vscale x 4 x i32>)
declare <vscale x 4 x float> @llvm.aarch64.sve.ld1.gather.uxtw.nxv4f32(<vscale x 4 x i1>, float*, <vscale x 4 x i32>)

; LD1D
declare <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.sxtw.nxv2i64(<vscale x 2 x i1>, i64*, <vscale x 2 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.uxtw.nxv2i64(<vscale x 2 x i1>, i64*, <vscale x 2 x i32>)

declare <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.sxtw.nxv2f64(<vscale x 2 x i1>, double*, <vscale x 2 x i32>)
declare <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.uxtw.nxv2f64(<vscale x 2 x i1>, double*, <vscale x 2 x i32>)

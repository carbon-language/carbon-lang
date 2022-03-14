; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; LDFF1H, LDFF1W, LDFF1D: base + 32-bit scaled offset, sign (sxtw) or zero (uxtw)
; extended to 64 bits
;   e.g. ldff1h z0.d, p0/z, [x0, z0.d, uxtw #1]
;

; LDFF1H
define <vscale x 4 x i32> @gldff1h_s_uxtw_index(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1h_s_uxtw_index:
; CHECK: ldff1h { z0.s }, p0/z, [x0, z0.s, uxtw #1]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv4i16(<vscale x 4 x i1> %pg,
                                                                                    i16* %base,
                                                                                    <vscale x 4 x i32> %b)
  %res = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @gldff1h_s_sxtw_index(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1h_s_sxtw_index:
; CHECK: ldff1h { z0.s }, p0/z, [x0, z0.s, sxtw #1]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv4i16(<vscale x 4 x i1> %pg,
                                                                                    i16* %base,
                                                                                    <vscale x 4 x i32> %b)
  %res = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldff1h_d_uxtw_index(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1h_d_uxtw_index:
; CHECK: ldff1h { z0.d }, p0/z, [x0, z0.d, uxtw #1]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv2i16(<vscale x 2 x i1> %pg,
                                                                                    i16* %base,
                                                                                    <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldff1h_d_sxtw_index(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1h_d_sxtw_index:
; CHECK: ldff1h { z0.d }, p0/z, [x0, z0.d, sxtw #1]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv2i16(<vscale x 2 x i1> %pg,
                                                                                    i16* %base,
                                                                                    <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDFF1W
define <vscale x 4 x i32> @gldff1w_s_uxtw_index(<vscale x 4 x i1> %pg, i32* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1w_s_uxtw_index:
; CHECK: ldff1w { z0.s }, p0/z, [x0, z0.s, uxtw #2]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                    i32* %base,
                                                                                    <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %load
}

define <vscale x 4 x i32> @gldff1w_s_sxtw_index(<vscale x 4 x i1> %pg, i32* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1w_s_sxtw_index:
; CHECK: ldff1w { z0.s }, p0/z, [x0, z0.s, sxtw #2]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                    i32* %base,
                                                                                    <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %load
}

define <vscale x 2 x i64> @gldff1w_d_uxtw_index(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1w_d_uxtw_index:
; CHECK: ldff1w { z0.d }, p0/z, [x0, z0.d, uxtw #2]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv2i32(<vscale x 2 x i1> %pg,
                                                                                    i32* %base,
                                                                                    <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldff1w_d_sxtw_index(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1w_d_sxtw_index:
; CHECK: ldff1w { z0.d }, p0/z, [x0, z0.d, sxtw #2]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv2i32(<vscale x 2 x i1> %pg,
                                                                                    i32* %base,
                                                                                    <vscale x 2 x i32> %b)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x float> @gldff1w_s_uxtw_index_float(<vscale x 4 x i1> %pg, float* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1w_s_uxtw_index_float:
; CHECK: ldff1w { z0.s }, p0/z, [x0, z0.s, uxtw #2]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv4f32(<vscale x 4 x i1> %pg,
                                                                                      float* %base,
                                                                                      <vscale x 4 x i32> %b)
  ret <vscale x 4 x float> %load
}

define <vscale x 4 x float> @gldff1w_s_sxtw_index_float(<vscale x 4 x i1> %pg, float* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1w_s_sxtw_index_float:
; CHECK: ldff1w { z0.s }, p0/z, [x0, z0.s, sxtw #2]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv4f32(<vscale x 4 x i1> %pg,
                                                                                      float* %base,
                                                                                      <vscale x 4 x i32> %b)
  ret <vscale x 4 x float> %load
}

; LDFF1D
define <vscale x 2 x i64> @gldff1d_s_uxtw_index(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1d_s_uxtw_index:
; CHECK: ldff1d { z0.d }, p0/z, [x0, z0.d, uxtw #3]
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                    i64* %base,
                                                                                    <vscale x 2 x i32> %b)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x i64> @gldff1d_sxtw_index(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1d_sxtw_index:
; CHECK: ldff1d { z0.d }, p0/z, [x0, z0.d, sxtw #3]
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                    i64* %base,
                                                                                    <vscale x 2 x i32> %b)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @gldff1d_uxtw_index_double(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1d_uxtw_index_double:
; CHECK: ldff1d { z0.d }, p0/z, [x0, z0.d, uxtw #3]
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv2f64(<vscale x 2 x i1> %pg,
                                                                                       double* %base,
                                                                                       <vscale x 2 x i32> %b)
  ret <vscale x 2 x double> %load
}

define <vscale x 2 x double> @gldff1d_sxtw_index_double(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1d_sxtw_index_double:
; CHECK: ldff1d { z0.d }, p0/z, [x0, z0.d, sxtw #3]
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv2f64(<vscale x 2 x i1> %pg,
                                                                                       double* %base,
                                                                                       <vscale x 2 x i32> %b)
  ret <vscale x 2 x double> %load
}

;
; LDFF1SH, LDFF1SW, LDFF1SD: base + 32-bit scaled offset, sign (sxtw) or zero (uxtw)
; extended to 64 bits
;   e.g. ldff1sh z0.d, p0/z, [x0, z0.d, uxtw #1]
;

; LDFF1SH
define <vscale x 4 x i32> @gldff1sh_s_uxtw_index(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1sh_s_uxtw_index:
; CHECK: ldff1sh { z0.s }, p0/z, [x0, z0.s, uxtw #1]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv4i16(<vscale x 4 x i1> %pg,
                                                                                    i16* %base,
                                                                                    <vscale x 4 x i32> %b)
  %res = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @gldff1sh_s_sxtw_index(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldff1sh_s_sxtw_index:
; CHECK: ldff1sh { z0.s }, p0/z, [x0, z0.s, sxtw #1]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv4i16(<vscale x 4 x i1> %pg,
                                                                                    i16* %base,
                                                                                    <vscale x 4 x i32> %b)
  %res = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldff1sh_d_uxtw_index(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1sh_d_uxtw_index:
; CHECK: ldff1sh { z0.d }, p0/z, [x0, z0.d, uxtw #1]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv2i16(<vscale x 2 x i1> %pg,
                                                                                    i16* %base,
                                                                                    <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldff1sh_d_sxtw_index(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1sh_d_sxtw_index:
; CHECK: ldff1sh { z0.d }, p0/z, [x0, z0.d, sxtw #1]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv2i16(<vscale x 2 x i1> %pg,
                                                                                    i16* %base,
                                                                                    <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDFF1SW
define <vscale x 2 x i64> @gldff1sw_d_uxtw_index(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1sw_d_uxtw_index:
; CHECK: ldff1sw { z0.d }, p0/z, [x0, z0.d, uxtw #2]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv2i32(<vscale x 2 x i1> %pg,
                                                                                    i32* %base,
                                                                                    <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gldff1sw_d_sxtw_index(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %b) {
; CHECK-LABEL: gldff1sw_d_sxtw_index:
; CHECK: ldff1sw { z0.d }, p0/z, [x0, z0.d, sxtw #2]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv2i32(<vscale x 2 x i1> %pg,
                                                                                    i32* %base,
                                                                                    <vscale x 2 x i32> %b)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}


; LDFF1H/LDFF1SH
declare <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv4i16(<vscale x 4 x i1>, i16*, <vscale x 4 x i32>)
declare <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv4i16(<vscale x 4 x i1>, i16*, <vscale x 4 x i32>)

declare <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv2i16(<vscale x 2 x i1>, i16*, <vscale x 2 x i32>)
declare <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv2i16(<vscale x 2 x i1>, i16*, <vscale x 2 x i32>)

; LDFF1W/LDFF1SW
declare <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv4i32(<vscale x 4 x i1>, i32*, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv4i32(<vscale x 4 x i1>, i32*, <vscale x 4 x i32>)

declare <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv2i32(<vscale x 2 x i1>, i32*, <vscale x 2 x i32>)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv2i32(<vscale x 2 x i1>, i32*, <vscale x 2 x i32>)

declare <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv4f32(<vscale x 4 x i1>, float*, <vscale x 4 x i32>)
declare <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv4f32(<vscale x 4 x i1>, float*, <vscale x 4 x i32>)

; LDFF1D
declare <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv2i64(<vscale x 2 x i1>, i64*, <vscale x 2 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv2i64(<vscale x 2 x i1>, i64*, <vscale x 2 x i32>)

declare <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv2f64(<vscale x 2 x i1>, double*, <vscale x 2 x i32>)
declare <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv2f64(<vscale x 2 x i1>, double*, <vscale x 2 x i32>)

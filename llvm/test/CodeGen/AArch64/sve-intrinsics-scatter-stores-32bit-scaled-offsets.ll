; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; ST1H, ST1W, ST1D: base + 32-bit scaled offset, sign (sxtw) or zero
; (uxtw) extended to 64 bits.
;   e.g. st1h { z0.d }, p0, [x0, z1.d, uxtw #1]
;

; ST1H
define void @sst1h_s_uxtw(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %indices) {
; CHECK-LABEL: sst1h_s_uxtw:
; CHECK: st1h { z0.s }, p0, [x0, z1.s, uxtw #1]
; CHECK-NEXT:	ret
  %data_trunc = trunc <vscale x 4 x i32> %data to <vscale x 4 x i16>
  call void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv4i16(<vscale x 4 x i16> %data_trunc,
                                                             <vscale x 4 x i1> %pg,
                                                             i16* %base,
                                                             <vscale x 4 x i32> %indices)
  ret void
}

define void @sst1h_s_sxtw(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %indices) {
; CHECK-LABEL: sst1h_s_sxtw:
; CHECK: st1h { z0.s }, p0, [x0, z1.s, sxtw #1]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 4 x i32> %data to <vscale x 4 x i16>
  call void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv4i16(<vscale x 4 x i16> %data_trunc,
                                                             <vscale x 4 x i1> %pg,
                                                             i16* %base,
                                                             <vscale x 4 x i32> %indices)
  ret void
}

define void @sst1h_d_uxtw(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %indices) {
; CHECK-LABEL: sst1h_d_uxtw:
; CHECK: st1h { z0.d }, p0, [x0, z1.d, uxtw #1]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i16>
  call void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv2i16(<vscale x 2 x i16> %data_trunc,
                                                             <vscale x 2 x i1> %pg,
                                                             i16* %base,
                                                             <vscale x 2 x i32> %indices)
  ret void
}

define void @sst1h_d_sxtw(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i32> %indices) {
; CHECK-LABEL: sst1h_d_sxtw:
; CHECK: st1h { z0.d }, p0, [x0, z1.d, sxtw #1]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i16>
  call void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv2i16(<vscale x 2 x i16> %data_trunc,
                                                             <vscale x 2 x i1> %pg,
                                                             i16* %base,
                                                             <vscale x 2 x i32> %indices)
  ret void
}

; ST1W
define void @sst1w_s_uxtw(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i32* %base, <vscale x 4 x i32> %indices) {
; CHECK-LABEL: sst1w_s_uxtw:
; CHECK: st1w { z0.s }, p0, [x0, z1.s, uxtw #2]
; CHECK-NEXT:	ret
  call void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv4i32(<vscale x 4 x i32> %data,
                                                             <vscale x 4 x i1> %pg,
                                                             i32* %base,
                                                             <vscale x 4 x i32> %indices)
  ret void
}

define void @sst1w_s_sxtw(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i32* %base, <vscale x 4 x i32> %indices) {
; CHECK-LABEL: sst1w_s_sxtw:
; CHECK: st1w { z0.s }, p0, [x0, z1.s, sxtw #2]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv4i32(<vscale x 4 x i32> %data,
                                                             <vscale x 4 x i1> %pg,
                                                             i32* %base,
                                                             <vscale x 4 x i32> %indices)
  ret void
}

define void @sst1w_d_uxtw(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %indices) {
; CHECK-LABEL: sst1w_d_uxtw:
; CHECK: st1w { z0.d }, p0, [x0, z1.d, uxtw #2]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  call void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv2i32(<vscale x 2 x i32> %data_trunc,
                                                             <vscale x 2 x i1> %pg,
                                                             i32* %base,
                                                             <vscale x 2 x i32> %indices)
  ret void
}

define void @sst1w_d_sxtw(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i32> %indices) {
; CHECK-LABEL: sst1w_d_sxtw:
; CHECK: st1w { z0.d }, p0, [x0, z1.d, sxtw #2]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  call void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv2i32(<vscale x 2 x i32> %data_trunc,
                                                             <vscale x 2 x i1> %pg,
                                                             i32* %base,
                                                             <vscale x 2 x i32> %indices)
  ret void
}

define void @sst1w_s_uxtw_float(<vscale x 4 x float> %data, <vscale x 4 x i1> %pg, float* %base, <vscale x 4 x i32> %indices) {
; CHECK-LABEL: sst1w_s_uxtw_float:
; CHECK: st1w { z0.s }, p0, [x0, z1.s, uxtw #2]
; CHECK-NEXT:	ret
  call void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv4f32(<vscale x 4 x float> %data,
                                                             <vscale x 4 x i1> %pg,
                                                             float* %base,
                                                             <vscale x 4 x i32> %indices)
  ret void
}

define void @sst1w_s_sxtw_float(<vscale x 4 x float> %data, <vscale x 4 x i1> %pg, float* %base, <vscale x 4 x i32> %indices) {
; CHECK-LABEL: sst1w_s_sxtw_float:
; CHECK: st1w { z0.s }, p0, [x0, z1.s, sxtw #2]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv4f32(<vscale x 4 x float> %data,
                                                             <vscale x 4 x i1> %pg,
                                                             float* %base,
                                                             <vscale x 4 x i32> %indices)
  ret void
}

; ST1D
define void @sst1d_d_uxtw(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i32> %indices) {
; CHECK-LABEL: sst1d_d_uxtw:
; CHECK: st1d { z0.d }, p0, [x0, z1.d, uxtw #3]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv2i64(<vscale x 2 x i64> %data,
                                                             <vscale x 2 x i1> %pg,
                                                             i64* %base,
                                                             <vscale x 2 x i32> %indices)
  ret void
}

define void @sst1d_d_sxtw(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i32> %indices) {
; CHECK-LABEL: sst1d_d_sxtw:
; CHECK: st1d { z0.d }, p0, [x0, z1.d, sxtw #3]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv2i64(<vscale x 2 x i64> %data,
                                                             <vscale x 2 x i1> %pg,
                                                             i64* %base,
                                                             <vscale x 2 x i32> %indices)
  ret void
}

define void @sst1d_d_uxtw_double(<vscale x 2 x double> %data, <vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i32> %indices) {
; CHECK-LABEL: sst1d_d_uxtw_double:
; CHECK: st1d { z0.d }, p0, [x0, z1.d, uxtw #3]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv2f64(<vscale x 2 x double> %data,
                                                             <vscale x 2 x i1> %pg,
                                                             double* %base,
                                                             <vscale x 2 x i32> %indices)
  ret void
}

define void @sst1d_d_sxtw_double(<vscale x 2 x double> %data, <vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i32> %indices) {
; CHECK-LABEL: sst1d_d_sxtw_double:
; CHECK: st1d { z0.d }, p0, [x0, z1.d, sxtw #3]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv2f64(<vscale x 2 x double> %data,
                                                             <vscale x 2 x i1> %pg,
                                                             double* %base,
                                                             <vscale x 2 x i32> %indices)
  ret void
}


; ST1H
declare void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv4i16(<vscale x 4 x i16>, <vscale x 4 x i1>, i16*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv2i16(<vscale x 2 x i16>, <vscale x 2 x i1>, i16*, <vscale x 2 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv4i16(<vscale x 4 x i16>, <vscale x 4 x i1>, i16*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv2i16(<vscale x 2 x i16>, <vscale x 2 x i1>, i16*, <vscale x 2 x i32>)

; ST1W
declare void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, i32*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i1>, i32*, <vscale x 2 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, i32*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i1>, i32*, <vscale x 2 x i32>)

declare void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, float*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, float*, <vscale x 4 x i32>)

; ST1D
declare void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, i64*, <vscale x 2 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, i64*, <vscale x 2 x i32>)

declare void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, double*, <vscale x 2 x i32>)
declare void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, double*, <vscale x 2 x i32>)

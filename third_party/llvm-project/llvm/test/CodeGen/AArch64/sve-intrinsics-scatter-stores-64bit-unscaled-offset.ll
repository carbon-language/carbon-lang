; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; ST1B, ST1W, ST1H, ST1D: base + 64-bit unscaled offset
;   e.g. st1h { z0.d }, p0, [x0, z1.d]
;

define void @sst1b_d(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i8* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sst1b_d:
; CHECK: st1b { z0.d }, p0, [x0, z1.d]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  call void @llvm.aarch64.sve.st1.scatter.nxv2i8(<vscale x 2 x i8> %data_trunc,
                                                 <vscale x 2 x i1> %pg,
                                                 i8* %base,
                                                 <vscale x 2 x i64> %b)
  ret void
}

define void @sst1h_d(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sst1h_d:
; CHECK: st1h { z0.d }, p0, [x0, z1.d]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i16>
  call void @llvm.aarch64.sve.st1.scatter.nxv2i16(<vscale x 2 x i16> %data_trunc,
                                                 <vscale x 2 x i1> %pg,
                                                 i16* %base,
                                                 <vscale x 2 x i64> %b)
  ret void
}

define void @sst1w_d(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sst1w_d:
; CHECK: st1w { z0.d }, p0, [x0, z1.d]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  call void @llvm.aarch64.sve.st1.scatter.nxv2i32(<vscale x 2 x i32> %data_trunc,
                                                 <vscale x 2 x i1> %pg,
                                                 i32* %base,
                                                 <vscale x 2 x i64> %b)
  ret void
}

define void @sst1d_d(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sst1d_d:
; CHECK: st1d { z0.d }, p0, [x0, z1.d]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.nxv2i64(<vscale x 2 x i64> %data,
                                                 <vscale x 2 x i1> %pg,
                                                 i64* %base,
                                                 <vscale x 2 x i64> %b)
  ret void
}

define void @sst1d_d_double(<vscale x 2 x double> %data, <vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sst1d_d_double:
; CHECK: st1d { z0.d }, p0, [x0, z1.d]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.st1.scatter.nxv2f64(<vscale x 2 x double> %data,
                                                 <vscale x 2 x i1> %pg,
                                                 double* %base,
                                                 <vscale x 2 x i64> %b)
  ret void
}

declare void @llvm.aarch64.sve.st1.scatter.nxv2i8(<vscale x 2 x i8>, <vscale x 2 x i1>, i8*, <vscale x 2 x i64>)
declare void @llvm.aarch64.sve.st1.scatter.nxv2i16(<vscale x 2 x i16>, <vscale x 2 x i1>, i16*, <vscale x 2 x i64>)
declare void @llvm.aarch64.sve.st1.scatter.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i1>, i32*, <vscale x 2 x i64>)
declare void @llvm.aarch64.sve.st1.scatter.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, i64*, <vscale x 2 x i64>)
declare void @llvm.aarch64.sve.st1.scatter.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, double*, <vscale x 2 x i64>)

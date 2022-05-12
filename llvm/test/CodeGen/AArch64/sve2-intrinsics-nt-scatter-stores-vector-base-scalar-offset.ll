; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s

;
; STNT1B, STNT1W, STNT1H, STNT1D: vector base + scalar offset
;   stnt1b { z0.s }, p0/z, [z0.s, x0]
;

; STNT1B
define void @stnt1b_s(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %base, i64 %offset) {
; CHECK-LABEL: stnt1b_s:
; CHECK: stnt1b { z0.s }, p0, [z1.s, x0]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  call void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv4i8.nxv4i32(<vscale x 4 x i8> %data_trunc,
                                                                         <vscale x 4 x i1> %pg,
                                                                         <vscale x 4 x i32> %base,
                                                                         i64 %offset)
  ret void
}

define void @stnt1b_d(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %base, i64 %offset) {
; CHECK-LABEL: stnt1b_d:
; CHECK: stnt1b { z0.d }, p0, [z1.d, x0]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  call void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i8> %data_trunc,
                                                                         <vscale x 2 x i1> %pg,
                                                                         <vscale x 2 x i64> %base,
                                                                         i64 %offset)
  ret void
}

; STNT1H
define void @stnt1h_s(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %base, i64 %offset) {
; CHECK-LABEL: stnt1h_s:
; CHECK: stnt1h { z0.s }, p0, [z1.s, x0]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 4 x i32> %data to <vscale x 4 x i16>
  call void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv4i16.nxv4i32(<vscale x 4 x i16> %data_trunc,
                                                                          <vscale x 4 x i1> %pg,
                                                                          <vscale x 4 x i32> %base,
                                                                          i64 %offset)
  ret void
}

define void @stnt1h_d(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %base, i64 %offset) {
; CHECK-LABEL: stnt1h_d:
; CHECK: stnt1h { z0.d }, p0, [z1.d, x0]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i16>
  call void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv2i16.nxv2i64(<vscale x 2 x i16> %data_trunc,
                                                                          <vscale x 2 x i1> %pg,
                                                                          <vscale x 2 x i64> %base,
                                                                          i64 %offset)
  ret void
}

; STNT1W
define void @stnt1w_s(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %base, i64 %offset) {
; CHECK-LABEL: stnt1w_s:
; CHECK: stnt1w { z0.s }, p0, [z1.s, x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i32> %data,
                                                                          <vscale x 4 x i1> %pg,
                                                                          <vscale x 4 x i32> %base,
                                                                          i64 %offset)
  ret void
}

define void @stnt1w_f32_s(<vscale x 4 x float> %data, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %base, i64 %offset) {
; CHECK-LABEL: stnt1w_f32_s:
; CHECK: stnt1w { z0.s }, p0, [z1.s, x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv4f32.nxv4i32(<vscale x 4 x float> %data,
                                                                          <vscale x 4 x i1> %pg,
                                                                          <vscale x 4 x i32> %base,
                                                                          i64 %offset)
  ret void
}

define void @stnt1w_d(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %base, i64 %offset) {
; CHECK-LABEL: stnt1w_d:
; CHECK: stnt1w { z0.d }, p0, [z1.d, x0]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  call void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i32> %data_trunc,
                                                                          <vscale x 2 x i1> %pg,
                                                                          <vscale x 2 x i64> %base,
                                                                          i64 %offset)
  ret void
}

; STNT1D
define void @stnt1d_d(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %base, i64 %offset) {
; CHECK-LABEL: stnt1d_d:
; CHECK: stnt1d { z0.d }, p0, [z1.d, x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i64> %data,
                                                                          <vscale x 2 x i1> %pg,
                                                                          <vscale x 2 x i64> %base,
                                                                          i64 %offset)
  ret void
}

define void @stnt1d_f64_d(<vscale x 2 x double> %data, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %base, i64 %offset) {
; CHECK-LABEL: stnt1d_f64_d:
; CHECK: stnt1d { z0.d }, p0, [z1.d, x0]
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv2f64.nxv2i64(<vscale x 2 x double> %data,
                                                                          <vscale x 2 x i1> %pg,
                                                                          <vscale x 2 x i64> %base,
                                                                          i64 %offset)
  ret void
}

; STNT1B
declare void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i8>, <vscale x 2 x i1>, <vscale x 2 x i64>, i64)
declare void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv4i8.nxv4i32(<vscale x 4 x i8>, <vscale x 4 x i1>, <vscale x 4 x i32>, i64)

; STNT1H
declare void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv2i16.nxv2i64(<vscale x 2 x i16>, <vscale x 2 x i1>, <vscale x 2 x i64>, i64)
declare void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv4i16.nxv4i32(<vscale x 4 x i16>, <vscale x 4 x i1>, <vscale x 4 x i32>, i64)

; STNT1W
declare void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i32>, <vscale x 2 x i1>, <vscale x 2 x i64>, i64)
declare void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>, i64)

declare void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv4f32.nxv4i32(<vscale x 4 x float>, <vscale x 4 x i1>, <vscale x 4 x i32>, i64)

; STNT1D
declare void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>, i64)

declare void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv2f32.nxv2i64(<vscale x 2 x float>, <vscale x 2 x i1>, <vscale x 2 x i64>, i64)
declare void @llvm.aarch64.sve.stnt1.scatter.scalar.offset.nxv2f64.nxv2i64(<vscale x 2 x double>, <vscale x 2 x i1>, <vscale x 2 x i64>, i64)

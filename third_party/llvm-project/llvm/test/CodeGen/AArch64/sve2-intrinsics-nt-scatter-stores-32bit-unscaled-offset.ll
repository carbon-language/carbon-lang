; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s

;
; STNT1B, STNT1W, STNT1H, STNT1D: base + 32-bit unscaled offset, zero (uxtw)
; extended to 64 bits.
;   e.g. stnt1h { z0.d }, p0, [z1.d, x0]
;

; STNT1B
define void @sstnt1b_s_uxtw(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i8* %base, <vscale x 4 x i32> %offsets) {
; CHECK-LABEL: sstnt1b_s_uxtw:
; CHECK: stnt1b { z0.s }, p0, [z1.s, x0]
; CHECK-NEXT: ret
  %data_trunc = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  call void  @llvm.aarch64.sve.stnt1.scatter.uxtw.nxv4i8(<vscale x 4 x i8> %data_trunc,
                                                         <vscale x 4 x i1> %pg,
                                                         i8* %base,
                                                         <vscale x 4 x i32> %offsets)
  ret void
}

; STNT1H
define void @sstnt1h_s_uxtw(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %offsets) {
; CHECK-LABEL: sstnt1h_s_uxtw:
; CHECK: stnt1h { z0.s }, p0, [z1.s, x0]
; CHECK-NEXT:	ret
  %data_trunc = trunc <vscale x 4 x i32> %data to <vscale x 4 x i16>
  call void @llvm.aarch64.sve.stnt1.scatter.uxtw.nxv4i16(<vscale x 4 x i16> %data_trunc,
                                                         <vscale x 4 x i1> %pg,
                                                         i16* %base,
                                                         <vscale x 4 x i32> %offsets)
  ret void
}

; STNT1W
define void @sstnt1w_s_uxtw(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i32* %base, <vscale x 4 x i32> %offsets) {
; CHECK-LABEL: sstnt1w_s_uxtw:
; CHECK: stnt1w { z0.s }, p0, [z1.s, x0]
; CHECK-NEXT:	ret
  call void @llvm.aarch64.sve.stnt1.scatter.uxtw.nxv4i32(<vscale x 4 x i32> %data,
                                                         <vscale x 4 x i1> %pg,
                                                         i32* %base,
                                                         <vscale x 4 x i32> %offsets)
  ret void
}

define void @sstnt1w_s_uxtw_float(<vscale x 4 x float> %data, <vscale x 4 x i1> %pg, float* %base, <vscale x 4 x i32> %offsets) {
; CHECK-LABEL: sstnt1w_s_uxtw_float:
; CHECK: stnt1w { z0.s }, p0, [z1.s, x0]
; CHECK-NEXT:	ret
  call void @llvm.aarch64.sve.stnt1.scatter.uxtw.nxv4f32(<vscale x 4 x float> %data,
                                                         <vscale x 4 x i1> %pg,
                                                         float* %base,
                                                         <vscale x 4 x i32> %offsets)
  ret void
}

; STNT1B
declare void @llvm.aarch64.sve.stnt1.scatter.uxtw.nxv4i8(<vscale x 4 x i8>, <vscale x 4 x i1>, i8*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.stnt1.scatter.uxtw.nxv2i8(<vscale x 2 x i8>, <vscale x 2 x i1>, i8*, <vscale x 2 x i32>)
declare void @llvm.aarch64.sve.stnt1.scatter.sxtw.nxv4i8(<vscale x 4 x i8>, <vscale x 4 x i1>, i8*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.stnt1.scatter.sxtw.nxv2i8(<vscale x 2 x i8>, <vscale x 2 x i1>, i8*, <vscale x 2 x i32>)

; STNT1H
declare void @llvm.aarch64.sve.stnt1.scatter.sxtw.nxv4i16(<vscale x 4 x i16>, <vscale x 4 x i1>, i16*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.stnt1.scatter.sxtw.nxv2i16(<vscale x 2 x i16>, <vscale x 2 x i1>, i16*, <vscale x 2 x i32>)
declare void @llvm.aarch64.sve.stnt1.scatter.uxtw.nxv4i16(<vscale x 4 x i16>, <vscale x 4 x i1>, i16*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.stnt1.scatter.uxtw.nxv2i16(<vscale x 2 x i16>, <vscale x 2 x i1>, i16*, <vscale x 2 x i32>)

; STNT1W
declare void @llvm.aarch64.sve.stnt1.scatter.sxtw.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, i32*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.stnt1.scatter.sxtw.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i1>, i32*, <vscale x 2 x i32>)
declare void @llvm.aarch64.sve.stnt1.scatter.uxtw.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, i32*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.stnt1.scatter.uxtw.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i1>, i32*, <vscale x 2 x i32>)

declare void @llvm.aarch64.sve.stnt1.scatter.sxtw.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, float*, <vscale x 4 x i32>)
declare void @llvm.aarch64.sve.stnt1.scatter.uxtw.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, float*, <vscale x 4 x i32>)

; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme < %s | FileCheck %s

;
; SXTB
;

define <vscale x 8 x i16> @sxtb_i16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sxtb_i16:
; CHECK: sxtb z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sxtb.nxv8i16(<vscale x 8 x i16> %a,
                                                                <vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sxtb_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sxtb_i32:
; CHECK: sxtb z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sxtb.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sxtb_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sxtb_i64:
; CHECK: sxtb z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sxtb.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; SXTH
;

define <vscale x 4 x i32> @sxth_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sxth_i32:
; CHECK: sxth z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sxth.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sxth_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sxth_i64:
; CHECK: sxth z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sxth.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; SXTW
;

define <vscale x 2 x i64> @sxtw_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sxtw_i64:
; CHECK: sxtw z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; UXTB
;

define <vscale x 8 x i16> @uxtb_i16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uxtb_i16:
; CHECK: uxtb z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uxtb.nxv8i16(<vscale x 8 x i16> %a,
                                                                <vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uxtb_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uxtb_i32:
; CHECK: uxtb z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uxtb.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uxtb_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %b) {
; CHECK-LABEL: uxtb_i64:
; CHECK: uxtb z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uxtb.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; UXTH
;

define <vscale x 4 x i32> @uxth_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uxth_i32:
; CHECK: uxth z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uxth.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uxth_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %b) {
; CHECK-LABEL: uxth_i64:
; CHECK: uxth z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uxth.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; UXTW
;

define <vscale x 2 x i64> @uxtw_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %b) {
; CHECK-LABEL: uxtw_i64:
; CHECK: uxtw z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

declare <vscale x 8 x i16> @llvm.aarch64.sve.sxtb.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sxtb.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sxtb.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sxth.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sxth.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.uxtb.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uxtb.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uxtb.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uxth.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uxth.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)

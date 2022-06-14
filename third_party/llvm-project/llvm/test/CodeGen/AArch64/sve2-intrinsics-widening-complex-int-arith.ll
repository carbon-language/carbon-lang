; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme < %s | FileCheck %s

;
; SADDLBT
;

define <vscale x 8 x i16> @saddlbt_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: saddlbt_b:
; CHECK: saddlbt z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.saddlbt.nxv8i16(<vscale x 16 x i8> %a,
                                                              <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @saddlbt_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: saddlbt_h:
; CHECK: saddlbt z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.saddlbt.nxv4i32(<vscale x 8 x i16> %a,
                                                              <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @saddlbt_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: saddlbt_s:
; CHECK: saddlbt z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.saddlbt.nxv2i64(<vscale x 4 x i32> %a,
                                                              <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; SSUBLBT
;

define <vscale x 8 x i16> @ssublbt_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: ssublbt_b:
; CHECK: ssublbt z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.ssublbt.nxv8i16(<vscale x 16 x i8> %a,
                                                              <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @ssublbt_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: ssublbt_h:
; CHECK: ssublbt z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.ssublbt.nxv4i32(<vscale x 8 x i16> %a,
                                                              <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @ssublbt_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: ssublbt_s:
; CHECK: ssublbt z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.ssublbt.nxv2i64(<vscale x 4 x i32> %a,
                                                              <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

;
; SSUBLTB
;

define <vscale x 8 x i16> @ssubltb_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: ssubltb_b:
; CHECK: ssubltb z0.h, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.ssubltb.nxv8i16(<vscale x 16 x i8> %a,
                                                              <vscale x 16 x i8> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @ssubltb_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: ssubltb_h:
; CHECK: ssubltb z0.s, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.ssubltb.nxv4i32(<vscale x 8 x i16> %a,
                                                              <vscale x 8 x i16> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @ssubltb_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: ssubltb_s:
; CHECK: ssubltb z0.d, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.ssubltb.nxv2i64(<vscale x 4 x i32> %a,
                                                              <vscale x 4 x i32> %b)
  ret <vscale x 2 x i64> %out
}

declare <vscale x 8 x i16> @llvm.aarch64.sve.saddlbt.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.saddlbt.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.saddlbt.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.ssublbt.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ssublbt.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ssublbt.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.ssubltb.nxv8i16(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ssubltb.nxv4i32(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ssubltb.nxv2i64(<vscale x 4 x i32>, <vscale x 4 x i32>)

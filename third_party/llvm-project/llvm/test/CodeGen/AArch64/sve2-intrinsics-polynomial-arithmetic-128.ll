; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2-aes -asm-verbose=0 < %s | FileCheck %s

;
; PMULLB
;

define <vscale x 2 x i64> @pmullb_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: pmullb_i64:
; CHECK: pmullb z0.q, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.pmullb.pair.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; PMULLT
;

define <vscale x 2 x i64> @pmullt_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: pmullt_i64:
; CHECK: pmullt z0.q, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.pmullt.pair.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

declare <vscale x 2 x i64> @llvm.aarch64.sve.pmullb.pair.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 2 x i64> @llvm.aarch64.sve.pmullt.pair.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

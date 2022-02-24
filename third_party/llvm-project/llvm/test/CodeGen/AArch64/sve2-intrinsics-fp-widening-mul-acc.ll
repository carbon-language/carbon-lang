; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s

;
; FMLALB (Vectors)
;

define <vscale x 4 x float> @fmlalb_h(<vscale x 4 x float> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fmlalb_h:
; CHECK: fmlalb z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmlalb.nxv4f32(<vscale x 4 x float> %a,
                                                                    <vscale x 8 x half> %b,
                                                                    <vscale x 8 x half> %c)
  ret <vscale x 4 x float> %out
}

;
; FMLALB (Indexed)
;

define <vscale x 4 x float> @fmlalb_lane_h(<vscale x 4 x float> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fmlalb_lane_h:
; CHECK: fmlalb z0.s, z1.h, z2.h[0]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmlalb.lane.nxv4f32(<vscale x 4 x float> %a,
                                                                         <vscale x 8 x half> %b,
                                                                         <vscale x 8 x half> %c,
                                                                         i32 0)
  ret <vscale x 4 x float> %out
}

;
; FMLALT (Vectors)
;

define <vscale x 4 x float> @fmlalt_h(<vscale x 4 x float> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fmlalt_h:
; CHECK: fmlalt z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmlalt.nxv4f32(<vscale x 4 x float> %a,
                                                                    <vscale x 8 x half> %b,
                                                                    <vscale x 8 x half> %c)
  ret <vscale x 4 x float> %out
}

;
; FMLALT (Indexed)
;

define <vscale x 4 x float> @fmlalt_lane_h(<vscale x 4 x float> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fmlalt_lane_h:
; CHECK: fmlalt z0.s, z1.h, z2.h[1]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmlalt.lane.nxv4f32(<vscale x 4 x float> %a,
                                                                         <vscale x 8 x half> %b,
                                                                         <vscale x 8 x half> %c,
                                                                         i32 1)
  ret <vscale x 4 x float> %out
}

;
; FMLSLB (Vectors)
;

define <vscale x 4 x float> @fmlslb_h(<vscale x 4 x float> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fmlslb_h:
; CHECK: fmlslb z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmlslb.nxv4f32(<vscale x 4 x float> %a,
                                                                    <vscale x 8 x half> %b,
                                                                    <vscale x 8 x half> %c)
  ret <vscale x 4 x float> %out
}

;
; FMLSLB (Indexed)
;

define <vscale x 4 x float> @fmlslb_lane_h(<vscale x 4 x float> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fmlslb_lane_h:
; CHECK: fmlslb z0.s, z1.h, z2.h[2]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmlslb.lane.nxv4f32(<vscale x 4 x float> %a,
                                                                         <vscale x 8 x half> %b,
                                                                         <vscale x 8 x half> %c,
                                                                         i32 2)
  ret <vscale x 4 x float> %out
}

;
; FMLSLT (Vectors)
;

define <vscale x 4 x float> @fmlslt_h(<vscale x 4 x float> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fmlslt_h:
; CHECK: fmlslt z0.s, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmlslt.nxv4f32(<vscale x 4 x float> %a,
                                                                    <vscale x 8 x half> %b,
                                                                    <vscale x 8 x half> %c)
 ret <vscale x 4 x float> %out
}

;
; FMLSLT (Indexed)
;

define <vscale x 4 x float> @fmlslt_lane_h(<vscale x 4 x float> %a, <vscale x 8 x half> %b, <vscale x 8 x half> %c) {
; CHECK-LABEL: fmlslt_lane_h:
; CHECK: fmlslt z0.s, z1.h, z2.h[3]
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmlslt.lane.nxv4f32(<vscale x 4 x float> %a,
                                                                         <vscale x 8 x half> %b,
                                                                         <vscale x 8 x half> %c,
                                                                         i32 3)
  ret <vscale x 4 x float> %out
}

declare <vscale x 4 x float> @llvm.aarch64.sve.fmlalb.nxv4f32(<vscale x 4 x float>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmlalb.lane.nxv4f32(<vscale x 4 x float>, <vscale x 8 x half>, <vscale x 8 x half>, i32)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmlalt.nxv4f32(<vscale x 4 x float>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmlalt.lane.nxv4f32(<vscale x 4 x float>, <vscale x 8 x half>, <vscale x 8 x half>, i32)

declare <vscale x 4 x float> @llvm.aarch64.sve.fmlslb.nxv4f32(<vscale x 4 x float>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmlslb.lane.nxv4f32(<vscale x 4 x float>, <vscale x 8 x half>, <vscale x 8 x half>, i32)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmlslt.nxv4f32(<vscale x 4 x float>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmlslt.lane.nxv4f32(<vscale x 4 x float>, <vscale x 8 x half>, <vscale x 8 x half>, i32)

; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme < %s | FileCheck %s

;
; CADD
;

define <vscale x 16 x i8> @cadd_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cadd_b:
; CHECK: cadd z0.b, z0.b, z1.b, #90
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.cadd.x.nxv16i8(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  i32 90)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @cadd_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cadd_h:
; CHECK: cadd z0.h, z0.h, z1.h, #90
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.cadd.x.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b,
                                                                  i32 90)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @cadd_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cadd_s:
; CHECK: cadd z0.s, z0.s, z1.s, #270
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.cadd.x.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b,
                                                                  i32 270)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @cadd_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cadd_d:
; CHECK: cadd z0.d, z0.d, z1.d, #270
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.cadd.x.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 2 x i64> %b,
                                                                  i32 270)
  ret <vscale x 2 x i64> %out
}

;
; SQCADD
;

define <vscale x 16 x i8> @sqcadd_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: sqcadd_b:
; CHECK: sqcadd z0.b, z0.b, z1.b, #90
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqcadd.x.nxv16i8(<vscale x 16 x i8> %a,
                                                                    <vscale x 16 x i8> %b,
                                                                    i32 90)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqcadd_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqcadd_h:
; CHECK: sqcadd z0.h, z0.h, z1.h, #90
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqcadd.x.nxv8i16(<vscale x 8 x i16> %a,
                                                                    <vscale x 8 x i16> %b,
                                                                    i32 90)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqcadd_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqcadd_s:
; CHECK: sqcadd z0.s, z0.s, z1.s, #270
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqcadd.x.nxv4i32(<vscale x 4 x i32> %a,
                                                                    <vscale x 4 x i32> %b,
                                                                    i32 270)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqcadd_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqcadd_d:
; CHECK: sqcadd z0.d, z0.d, z1.d, #270
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqcadd.x.nxv2i64(<vscale x 2 x i64> %a,
                                                                    <vscale x 2 x i64> %b,
                                                                    i32 270)
  ret <vscale x 2 x i64> %out
}

;
; CMLA
;

define <vscale x 16 x i8> @cmla_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: cmla_b:
; CHECK: cmla z0.b, z1.b, z2.b, #90
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.cmla.x.nxv16i8(<vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b,
                                                                  <vscale x 16 x i8> %c,
                                                                  i32 90)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @cmla_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: cmla_h:
; CHECK: cmla z0.h, z1.h, z2.h, #180
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.cmla.x.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b,
                                                                  <vscale x 8 x i16> %c,
                                                                  i32 180)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @cmla_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: cmla_s:
; CHECK: cmla z0.s, z1.s, z2.s, #270
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.cmla.x.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b,
                                                                  <vscale x 4 x i32> %c,
                                                                  i32 270)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @cmla_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: cmla_d:
; CHECK: cmla z0.d, z1.d, z2.d, #0
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.cmla.x.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 2 x i64> %b,
                                                                  <vscale x 2 x i64> %c,
                                                                  i32 0)
  ret <vscale x 2 x i64> %out
}

;
; CMLA_LANE
;

define <vscale x 8 x i16> @cmla_lane_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: cmla_lane_h:
; CHECK: cmla z0.h, z1.h, z2.h[1], #180
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.cmla.lane.x.nxv8i16(<vscale x 8 x i16> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 1,
                                                                       i32 180)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @cmla_lane_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: cmla_lane_s:
; CHECK: cmla z0.s, z1.s, z2.s[0], #270
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.cmla.lane.x.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 0,
                                                                       i32 270)
  ret <vscale x 4 x i32> %out
}

;
; QRDCMLAH
;

define <vscale x 16 x i8> @sqrdcmlah_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: sqrdcmlah_b:
; CHECK: sqrdcmlah z0.b, z1.b, z2.b, #0
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqrdcmlah.x.nxv16i8(<vscale x 16 x i8> %a,
                                                                       <vscale x 16 x i8> %b,
                                                                       <vscale x 16 x i8> %c,
                                                                       i32 0)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqrdcmlah_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqrdcmlah_h:
; CHECK: sqrdcmlah z0.h, z1.h, z2.h, #90
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrdcmlah.x.nxv8i16(<vscale x 8 x i16> %a,
                                                                       <vscale x 8 x i16> %b,
                                                                       <vscale x 8 x i16> %c,
                                                                       i32 90)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqrdcmlah_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqrdcmlah_s:
; CHECK: sqrdcmlah z0.s, z1.s, z2.s, #180
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrdcmlah.x.nxv4i32(<vscale x 4 x i32> %a,
                                                                       <vscale x 4 x i32> %b,
                                                                       <vscale x 4 x i32> %c,
                                                                       i32 180)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqrdcmlah_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: sqrdcmlah_d:
; CHECK: sqrdcmlah z0.d, z1.d, z2.d, #270
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqrdcmlah.x.nxv2i64(<vscale x 2 x i64> %a,
                                                                       <vscale x 2 x i64> %b,
                                                                       <vscale x 2 x i64> %c,
                                                                       i32 270)
  ret <vscale x 2 x i64> %out
}

;
; QRDCMLAH_LANE
;

define <vscale x 8 x i16> @sqrdcmlah_lane_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: sqrdcmlah_lane_h:
; CHECK: sqrdcmlah z0.h, z1.h, z2.h[1], #90
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrdcmlah.lane.x.nxv8i16(<vscale x 8 x i16> %a,
                                                                            <vscale x 8 x i16> %b,
                                                                            <vscale x 8 x i16> %c,
                                                                            i32 1,
                                                                            i32 90)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqrdcmlah_lane_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: sqrdcmlah_lane_s:
; CHECK: sqrdcmlah z0.s, z1.s, z2.s[0], #180
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrdcmlah.lane.x.nxv4i32(<vscale x 4 x i32> %a,
                                                                            <vscale x 4 x i32> %b,
                                                                            <vscale x 4 x i32> %c,
                                                                            i32 0,
                                                                            i32 180)
  ret <vscale x 4 x i32> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.cadd.x.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.cadd.x.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.cadd.x.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.cadd.x.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqcadd.x.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqcadd.x.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqcadd.x.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqcadd.x.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.cmla.x.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.cmla.x.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.cmla.x.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.cmla.x.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, i32)

declare <vscale x 8 x i16> @llvm.aarch64.sve.cmla.lane.x.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.cmla.lane.x.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqrdcmlah.x.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqrdcmlah.x.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqrdcmlah.x.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqrdcmlah.x.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, i32)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sqrdcmlah.lane.x.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, i32, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqrdcmlah.lane.x.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, i32, i32)

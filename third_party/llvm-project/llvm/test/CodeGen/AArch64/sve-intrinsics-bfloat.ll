; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve,+bf16 -asm-verbose=0 < %s | FileCheck %s

;
; BFDOT
;

define <vscale x 4 x float> @bfdot_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfdot_f32:
; CHECK-NEXT:  bfdot z0.s, z1.h, z2.h
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfdot(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfdot_lane_0_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfdot_lane_0_f32:
; CHECK-NEXT:  bfdot z0.s, z1.h, z2.h[0]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfdot.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 0)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfdot_lane_1_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfdot_lane_1_f32:
; CHECK-NEXT:  bfdot z0.s, z1.h, z2.h[1]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfdot.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 1)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfdot_lane_2_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfdot_lane_2_f32:
; CHECK-NEXT:  bfdot z0.s, z1.h, z2.h[2]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfdot.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 2)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfdot_lane_3_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfdot_lane_3_f32:
; CHECK-NEXT:  bfdot z0.s, z1.h, z2.h[3]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfdot.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 3)
  ret <vscale x 4 x float> %out
}

;
; BFMLALB
;

define <vscale x 4 x float> @bfmlalb_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalb_f32:
; CHECK-NEXT:  bfmlalb z0.s, z1.h, z2.h
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalb(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfmlalb_lane_0_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalb_lane_0_f32:
; CHECK-NEXT:  bfmlalb z0.s, z1.h, z2.h[0]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalb.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 0)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfmlalb_lane_1_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalb_lane_1_f32:
; CHECK-NEXT:  bfmlalb z0.s, z1.h, z2.h[1]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalb.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 1)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfmlalb_lane_2_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalb_lane_2_f32:
; CHECK-NEXT:  bfmlalb z0.s, z1.h, z2.h[2]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalb.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 2)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfmlalb_lane_3_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalb_lane_3_f32:
; CHECK-NEXT:  bfmlalb z0.s, z1.h, z2.h[3]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalb.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 3)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfmlalb_lane_4_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalb_lane_4_f32:
; CHECK-NEXT:  bfmlalb z0.s, z1.h, z2.h[4]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalb.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 4)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfmlalb_lane_5_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalb_lane_5_f32:
; CHECK-NEXT:  bfmlalb z0.s, z1.h, z2.h[5]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalb.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 5)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfmlalb_lane_6_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalb_lane_6_f32:
; CHECK-NEXT:  bfmlalb z0.s, z1.h, z2.h[6]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalb.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 6)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfmlalb_lane_7_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalb_lane_7_f32:
; CHECK-NEXT:  bfmlalb z0.s, z1.h, z2.h[7]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalb.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 7)
  ret <vscale x 4 x float> %out
}

;
; BFMLALT
;

define <vscale x 4 x float> @bfmlalt_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalt_f32:
; CHECK-NEXT:  bfmlalt z0.s, z1.h, z2.h
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalt(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfmlalt_lane_0_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalt_lane_0_f32:
; CHECK-NEXT:  bfmlalt z0.s, z1.h, z2.h[0]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalt.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 0)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfmlalt_lane_1_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalt_lane_1_f32:
; CHECK-NEXT:  bfmlalt z0.s, z1.h, z2.h[1]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalt.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 1)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfmlalt_lane_2_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalt_lane_2_f32:
; CHECK-NEXT:  bfmlalt z0.s, z1.h, z2.h[2]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalt.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 2)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfmlalt_lane_3_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalt_lane_3_f32:
; CHECK-NEXT:  bfmlalt z0.s, z1.h, z2.h[3]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalt.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 3)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfmlalt_lane_4_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalt_lane_4_f32:
; CHECK-NEXT:  bfmlalt z0.s, z1.h, z2.h[4]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalt.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 4)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfmlalt_lane_5_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalt_lane_5_f32:
; CHECK-NEXT:  bfmlalt z0.s, z1.h, z2.h[5]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalt.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 5)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfmlalt_lane_6_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalt_lane_6_f32:
; CHECK-NEXT:  bfmlalt z0.s, z1.h, z2.h[6]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalt.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 6)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @bfmlalt_lane_7_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmlalt_lane_7_f32:
; CHECK-NEXT:  bfmlalt z0.s, z1.h, z2.h[7]
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalt.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 7)
  ret <vscale x 4 x float> %out
}

;
; BFMMLA
;

define <vscale x 4 x float> @bfmmla_f32(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: bfmmla_f32:
; CHECK-NEXT:  bfmmla z0.s, z1.h, z2.h
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmmla(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c)
  ret <vscale x 4 x float> %out
}

;
; BFCVT
;

define <vscale x 8 x bfloat> @fcvt_bf16_f32(<vscale x 8 x bfloat> %a, <vscale x 8 x i1> %pg, <vscale x 4 x float> %b) nounwind {
; CHECK-LABEL: fcvt_bf16_f32:
; CHECK-NEXT: bfcvt z0.h, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x bfloat> @llvm.aarch64.sve.fcvt.bf16f32(<vscale x 8 x bfloat> %a, <vscale x 8 x i1> %pg, <vscale x 4 x float> %b)
  ret <vscale x 8 x bfloat> %out
}

;
; BFCVTNT
;

define <vscale x 8 x bfloat> @fcvtnt_bf16_f32(<vscale x 8 x bfloat> %a, <vscale x 8 x i1> %pg, <vscale x 4 x float> %b) nounwind {
; CHECK-LABEL: fcvtnt_bf16_f32:
; CHECK-NEXT: bfcvtnt z0.h, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x bfloat> @llvm.aarch64.sve.fcvtnt.bf16f32(<vscale x 8 x bfloat> %a, <vscale x 8 x i1> %pg, <vscale x 4 x float> %b)
  ret <vscale x 8 x bfloat> %out
}

declare <vscale x 4 x float> @llvm.aarch64.sve.bfdot(<vscale x 4 x float>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>)
declare <vscale x 4 x float> @llvm.aarch64.sve.bfdot.lane(<vscale x 4 x float>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, i64)
declare <vscale x 4 x float> @llvm.aarch64.sve.bfmlalb(<vscale x 4 x float>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>)
declare <vscale x 4 x float> @llvm.aarch64.sve.bfmlalb.lane(<vscale x 4 x float>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, i64)
declare <vscale x 4 x float> @llvm.aarch64.sve.bfmlalt(<vscale x 4 x float>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>)
declare <vscale x 4 x float> @llvm.aarch64.sve.bfmlalt.lane(<vscale x 4 x float>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, i64)
declare <vscale x 4 x float> @llvm.aarch64.sve.bfmmla(<vscale x 4 x float>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.fcvt.bf16f32(<vscale x 8 x bfloat>, <vscale x 8 x i1>, <vscale x 4 x float>)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.fcvtnt.bf16f32(<vscale x 8 x bfloat>, <vscale x 8 x i1>, <vscale x 4 x float>)

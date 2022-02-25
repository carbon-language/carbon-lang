; RUN: llc -aarch64-sve-vector-bits-min=128  < %s | FileCheck %s -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_EQ_256
; RUN: llc -aarch64-sve-vector-bits-min=384  < %s | FileCheck %s -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=512  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_GE_2048

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

;
; UCVTF H -> H
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @ucvtf_v4i16_v4f16(<4 x i16> %op1) #0 {
; CHECK-LABEL: ucvtf_v4i16_v4f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ucvtf v0.4h, v0.4h
; CHECK-NEXT:    ret
  %res = uitofp <4 x i16> %op1 to <4 x half>
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define void @ucvtf_v8i16_v8f16(<8 x i16>* %a, <8 x half>* %b) #0 {
; CHECK-LABEL: ucvtf_v8i16_v8f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x0]
; CHECK-NEXT:    ucvtf v0.8h, v0.8h
; CHECK-NEXT:    str q0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <8 x i16>, <8 x i16>* %a
  %res = uitofp <8 x i16> %op1 to <8 x half>
  store <8 x half> %res, <8 x half>* %b
  ret void
}

define void @ucvtf_v16i16_v16f16(<16 x i16>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: ucvtf_v16i16_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ucvtf z0.h, p0/m, z0.h
; CHECK-NEXT:    st1h { z0.h }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %res = uitofp <16 x i16> %op1 to <16 x half>
  store <16 x half> %res, <16 x half>* %b
  ret void
}

define void @ucvtf_v32i16_v32f16(<32 x i16>* %a, <32 x half>* %b) #0 {
; VBITS_EQ_256-LABEL: ucvtf_v32i16_v32f16:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #16
; VBITS_EQ_256-NEXT:    ptrue p0.h, vl16
; VBITS_EQ_256-NEXT:    ld1h { z0.h }, p0/z, [x0, x8, lsl #1]
; VBITS_EQ_256-NEXT:    ld1h { z1.h }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ucvtf z0.h, p0/m, z0.h
; VBITS_EQ_256-NEXT:    ucvtf z1.h, p0/m, z1.h
; VBITS_EQ_256-NEXT:    st1h { z0.h }, p0, [x1, x8, lsl #1]
; VBITS_EQ_256-NEXT:    st1h { z1.h }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: ucvtf_v32i16_v32f16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl32
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ucvtf z0.h, p0/m, z0.h
; VBITS_GE_512-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %res = uitofp <32 x i16> %op1 to <32 x half>
  store <32 x half> %res, <32 x half>* %b
  ret void
}

define void @ucvtf_v64i16_v64f16(<64 x i16>* %a, <64 x half>* %b) #0 {
; VBITS_GE_1024-LABEL: ucvtf_v64i16_v64f16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl64
; VBITS_GE_1024-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ucvtf z0.h, p0/m, z0.h
; VBITS_GE_1024-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %res = uitofp <64 x i16> %op1 to <64 x half>
  store <64 x half> %res, <64 x half>* %b
  ret void
}

define void @ucvtf_v128i16_v128f16(<128 x i16>* %a, <128 x half>* %b) #0 {
; VBITS_GE_2048-LABEL: ucvtf_v128i16_v128f16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl128
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ucvtf z0.h, p0/m, z0.h
; VBITS_GE_2048-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %res = uitofp <128 x i16> %op1 to <128 x half>
  store <128 x half> %res, <128 x half>* %b
  ret void
}

;
; UCVTF H -> S
;

; Don't use SVE for 64-bit vectors.
define <2 x float> @ucvtf_v2i16_v2f32(<2 x i16> %op1) #0 {
; CHECK-LABEL: ucvtf_v2i16_v2f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movi d1, #0x00ffff0000ffff
; CHECK-NEXT:    and v0.8b, v0.8b, v1.8b
; CHECK-NEXT:    ucvtf v0.2s, v0.2s
; CHECK-NEXT:    ret
  %res = uitofp <2 x i16> %op1 to <2 x float>
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @ucvtf_v4i16_v4f32(<4 x i16> %op1) #0 {
; CHECK-LABEL: ucvtf_v4i16_v4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ushll v0.4s, v0.4h, #0
; CHECK-NEXT:    ucvtf v0.4s, v0.4s
; CHECK-NEXT:    ret
  %res = uitofp <4 x i16> %op1 to <4 x float>
  ret <4 x float> %res
}

define void @ucvtf_v8i16_v8f32(<8 x i16>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: ucvtf_v8i16_v8f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x0]
; CHECK-NEXT:    ptrue p0.s, vl8
; CHECK-NEXT:    uunpklo z0.s, z0.h
; CHECK-NEXT:    ucvtf z0.s, p0/m, z0.s
; CHECK-NEXT:    st1w { z0.s }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <8 x i16>, <8 x i16>* %a
  %res = uitofp <8 x i16> %op1 to <8 x float>
  store <8 x float> %res, <8 x float>* %b
  ret void
}

define void @ucvtf_v16i16_v16f32(<16 x i16>* %a, <16 x float>* %b) #0 {
; VBITS_EQ_256-LABEL: ucvtf_v16i16_v16f32:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    ptrue p0.h, vl16
; VBITS_EQ_256-NEXT:    mov x8, #8
; VBITS_EQ_256-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    uunpklo z1.s, z0.h
; VBITS_EQ_256-NEXT:    ext z0.b, z0.b, z0.b, #16
; VBITS_EQ_256-NEXT:    uunpklo z0.s, z0.h
; VBITS_EQ_256-NEXT:    ucvtf z1.s, p0/m, z1.s
; VBITS_EQ_256-NEXT:    ucvtf z0.s, p0/m, z0.s
; VBITS_EQ_256-NEXT:    st1w { z1.s }, p0, [x1]
; VBITS_EQ_256-NEXT:    st1w { z0.s }, p0, [x1, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: ucvtf_v16i16_v16f32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl16
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_512-NEXT:    ucvtf z0.s, p0/m, z0.s
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %res = uitofp <16 x i16> %op1 to <16 x float>
  store <16 x float> %res, <16 x float>* %b
  ret void
}

define void @ucvtf_v32i16_v32f32(<32 x i16>* %a, <32 x float>* %b) #0 {
; VBITS_GE_1024-LABEL: ucvtf_v32i16_v32f32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl32
; VBITS_GE_1024-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_1024-NEXT:    ucvtf z0.s, p0/m, z0.s
; VBITS_GE_1024-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %res = uitofp <32 x i16> %op1 to <32 x float>
  store <32 x float> %res, <32 x float>* %b
  ret void
}

define void @ucvtf_v64i16_v64f32(<64 x i16>* %a, <64 x float>* %b) #0 {
; VBITS_GE_2048-LABEL: ucvtf_v64i16_v64f32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl64
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_2048-NEXT:    ucvtf z0.s, p0/m, z0.s
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %res = uitofp <64 x i16> %op1 to <64 x float>
  store <64 x float> %res, <64 x float>* %b
  ret void
}

;
; UCVTF H -> D
;

; v1i16 is perfered to be widened to v4i16, which pushes the output into SVE types, so use SVE
define <1 x double> @ucvtf_v1i16_v1f64(<1 x i16> %op1) #0 {
; CHECK-LABEL: ucvtf_v1i16_v1f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $z0
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    uunpklo z0.s, z0.h
; CHECK-NEXT:    uunpklo z0.d, z0.s
; CHECK-NEXT:    ucvtf z0.d, p0/m, z0.d
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $z0
; CHECK-NEXT:    ret
  %res = uitofp <1 x i16> %op1 to <1 x double>
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @ucvtf_v2i16_v2f64(<2 x i16> %op1) #0 {
; CHECK-LABEL: ucvtf_v2i16_v2f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movi d1, #0x00ffff0000ffff
; CHECK-NEXT:    and v0.8b, v0.8b, v1.8b
; CHECK-NEXT:    ushll v0.2d, v0.2s, #0
; CHECK-NEXT:    ucvtf v0.2d, v0.2d
; CHECK-NEXT:    ret
  %res = uitofp <2 x i16> %op1 to <2 x double>
  ret <2 x double> %res
}

define void @ucvtf_v4i16_v4f64(<4 x i16>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: ucvtf_v4i16_v4f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr d0, [x0]
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    uunpklo z0.s, z0.h
; CHECK-NEXT:    uunpklo z0.d, z0.s
; CHECK-NEXT:    ucvtf z0.d, p0/m, z0.d
; CHECK-NEXT:    st1d { z0.d }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <4 x i16>, <4 x i16>* %a
  %res = uitofp <4 x i16> %op1 to <4 x double>
  store <4 x double> %res, <4 x double>* %b
  ret void
}

define void @ucvtf_v8i16_v8f64(<8 x i16>* %a, <8 x double>* %b) #0 {
; VBITS_EQ_256-LABEL: ucvtf_v8i16_v8f64:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    ldr q0, [x0]
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ext v1.16b, v0.16b, v0.16b, #8
; VBITS_EQ_256-NEXT:    uunpklo z0.s, z0.h
; VBITS_EQ_256-NEXT:    uunpklo z0.d, z0.s
; VBITS_EQ_256-NEXT:    ucvtf z0.d, p0/m, z0.d
; VBITS_EQ_256-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_EQ_256-NEXT:    uunpklo z1.s, z1.h
; VBITS_EQ_256-NEXT:    uunpklo z1.d, z1.s
; VBITS_EQ_256-NEXT:    ucvtf z1.d, p0/m, z1.d
; VBITS_EQ_256-NEXT:    st1d { z1.d }, p0, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: ucvtf_v8i16_v8f64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ldr q0, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_512-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_512-NEXT:    ucvtf z0.d, p0/m, z0.d
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x i16>, <8 x i16>* %a
  %res = uitofp <8 x i16> %op1 to <8 x double>
  store <8 x double> %res, <8 x double>* %b
  ret void
}

define void @ucvtf_v16i16_v16f64(<16 x i16>* %a, <16 x double>* %b) #0 {
; VBITS_GE_1024-LABEL: ucvtf_v16i16_v16f64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl16
; VBITS_GE_1024-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_1024-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_1024-NEXT:    ucvtf z0.d, p0/m, z0.d
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %res = uitofp <16 x i16> %op1 to <16 x double>
  store <16 x double> %res, <16 x double>* %b
  ret void
}

define void @ucvtf_v32i16_v32f64(<32 x i16>* %a, <32 x double>* %b) #0 {
; VBITS_GE_2048-LABEL: ucvtf_v32i16_v32f64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl32
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_2048-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    ucvtf z0.d, p0/m, z0.d
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %res = uitofp <32 x i16> %op1 to <32 x double>
  store <32 x double> %res, <32 x double>* %b
  ret void
}

;
; UCVTF S -> H
;

; Don't use SVE for 64-bit vectors.
define <2 x half> @ucvtf_v2i32_v2f16(<2 x i32> %op1) #0 {
; CHECK-LABEL: ucvtf_v2i32_v2f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $q0
; CHECK-NEXT:    ucvtf v0.4s, v0.4s
; CHECK-NEXT:    fcvtn v0.4h, v0.4s
; CHECK-NEXT:    ret
  %res = uitofp <2 x i32> %op1 to <2 x half>
  ret <2 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x half> @ucvtf_v4i32_v4f16(<4 x i32> %op1) #0 {
; CHECK-LABEL: ucvtf_v4i32_v4f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ucvtf v0.4s, v0.4s
; CHECK-NEXT:    fcvtn v0.4h, v0.4s
; CHECK-NEXT:    ret
  %res = uitofp <4 x i32> %op1 to <4 x half>
  ret <4 x half> %res
}

define <8 x half> @ucvtf_v8i32_v8f16(<8 x i32>* %a) #0 {
; CHECK-LABEL: ucvtf_v8i32_v8f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s, vl8
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    ucvtf z0.h, p0/m, z0.s
; CHECK-NEXT:    uzp1 z0.h, z0.h, z0.h
; CHECK-NEXT:    // kill: def $q0 killed $q0 killed $z0
; CHECK-NEXT:    ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %res = uitofp <8 x i32> %op1 to <8 x half>
  ret <8 x half> %res
}

define void @ucvtf_v16i32_v16f16(<16 x i32>* %a, <16 x half>* %b) #0 {
; VBITS_EQ_256-LABEL: ucvtf_v16i32_v16f16:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #8
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    ld1w { z0.s }, p0/z, [x0, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ld1w { z1.s }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ptrue p0.s
; VBITS_EQ_256-NEXT:    ucvtf z0.h, p0/m, z0.s
; VBITS_EQ_256-NEXT:    ucvtf z1.h, p0/m, z1.s
; VBITS_EQ_256-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_EQ_256-NEXT:    uzp1 z1.h, z1.h, z1.h
; VBITS_EQ_256-NEXT:    ptrue p0.h, vl8
; VBITS_EQ_256-NEXT:    splice z1.h, p0, z1.h, z0.h
; VBITS_EQ_256-NEXT:    ptrue p0.h, vl16
; VBITS_EQ_256-NEXT:    st1h { z1.h }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: ucvtf_v16i32_v16f16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.s
; VBITS_GE_512-NEXT:    ucvtf z0.h, p0/m, z0.s
; VBITS_GE_512-NEXT:    ptrue p0.h, vl16
; VBITS_GE_512-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_512-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %res = uitofp <16 x i32> %op1 to <16 x half>
  store <16 x half> %res, <16 x half>* %b
  ret void
}

define void @ucvtf_v32i32_v32f16(<32 x i32>* %a, <32 x half>* %b) #0 {
; VBITS_GE_1024-LABEL: ucvtf_v32i32_v32f16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.s
; VBITS_GE_1024-NEXT:    ucvtf z0.h, p0/m, z0.s
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl32
; VBITS_GE_1024-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_1024-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %res = uitofp <32 x i32> %op1 to <32 x half>
  store <32 x half> %res, <32 x half>* %b
  ret void
}

define void @ucvtf_v64i32_v64f16(<64 x i32>* %a, <64 x half>* %b) #0 {
; VBITS_GE_2048-LABEL: ucvtf_v64i32_v64f16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.s
; VBITS_GE_2048-NEXT:    ucvtf z0.h, p0/m, z0.s
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl64
; VBITS_GE_2048-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_2048-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %res = uitofp <64 x i32> %op1 to <64 x half>
  store <64 x half> %res, <64 x half>* %b
  ret void
}

;
; UCVTF S -> S
;

; Don't use SVE for 64-bit vectors.
define <2 x float> @ucvtf_v2i32_v2f32(<2 x i32> %op1) #0 {
; CHECK-LABEL: ucvtf_v2i32_v2f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ucvtf v0.2s, v0.2s
; CHECK-NEXT:    ret
  %res = uitofp <2 x i32> %op1 to <2 x float>
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @ucvtf_v4i32_v4f32(<4 x i32> %op1) #0 {
; CHECK-LABEL: ucvtf_v4i32_v4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ucvtf v0.4s, v0.4s
; CHECK-NEXT:    ret
  %res = uitofp <4 x i32> %op1 to <4 x float>
  ret <4 x float> %res
}

define void @ucvtf_v8i32_v8f32(<8 x i32>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: ucvtf_v8i32_v8f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s, vl8
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    ucvtf z0.s, p0/m, z0.s
; CHECK-NEXT:    st1w { z0.s }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %res = uitofp <8 x i32> %op1 to <8 x float>
  store <8 x float> %res, <8 x float>* %b
  ret void
}

define void @ucvtf_v16i32_v16f32(<16 x i32>* %a, <16 x float>* %b) #0 {
; VBITS_EQ_256-LABEL: ucvtf_v16i32_v16f32:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #8
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    ld1w { z0.s }, p0/z, [x0, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ld1w { z1.s }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ucvtf z0.s, p0/m, z0.s
; VBITS_EQ_256-NEXT:    ucvtf z1.s, p0/m, z1.s
; VBITS_EQ_256-NEXT:    st1w { z0.s }, p0, [x1, x8, lsl #2]
; VBITS_EQ_256-NEXT:    st1w { z1.s }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: ucvtf_v16i32_v16f32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ucvtf z0.s, p0/m, z0.s
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %res = uitofp <16 x i32> %op1 to <16 x float>
  store <16 x float> %res, <16 x float>* %b
  ret void
}

define void @ucvtf_v32i32_v32f32(<32 x i32>* %a, <32 x float>* %b) #0 {
; VBITS_GE_1024-LABEL: ucvtf_v32i32_v32f32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ucvtf z0.s, p0/m, z0.s
; VBITS_GE_1024-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %res = uitofp <32 x i32> %op1 to <32 x float>
  store <32 x float> %res, <32 x float>* %b
  ret void
}

define void @ucvtf_v64i32_v64f32(<64 x i32>* %a, <64 x float>* %b) #0 {
; VBITS_GE_2048-LABEL: ucvtf_v64i32_v64f32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ucvtf z0.s, p0/m, z0.s
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %res = uitofp <64 x i32> %op1 to <64 x float>
  store <64 x float> %res, <64 x float>* %b
  ret void
}

;
; UCVTF S -> D
;

; Don't use SVE for 64-bit vectors.
define <1 x double> @ucvtf_v1i32_v1f64(<1 x i32> %op1) #0 {
; CHECK-LABEL: ucvtf_v1i32_v1f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ushll v0.2d, v0.2s, #0
; CHECK-NEXT:    ucvtf v0.2d, v0.2d
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $q0
; CHECK-NEXT:    ret
  %res = uitofp <1 x i32> %op1 to <1 x double>
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @ucvtf_v2i32_v2f64(<2 x i32> %op1) #0 {
; CHECK-LABEL: ucvtf_v2i32_v2f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ushll v0.2d, v0.2s, #0
; CHECK-NEXT:    ucvtf v0.2d, v0.2d
; CHECK-NEXT:    ret
  %res = uitofp <2 x i32> %op1 to <2 x double>
  ret <2 x double> %res
}

define void @ucvtf_v4i32_v4f64(<4 x i32>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: ucvtf_v4i32_v4f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x0]
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    uunpklo z0.d, z0.s
; CHECK-NEXT:    ucvtf z0.d, p0/m, z0.d
; CHECK-NEXT:    st1d { z0.d }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <4 x i32>, <4 x i32>* %a
  %res = uitofp <4 x i32> %op1 to <4 x double>
  store <4 x double> %res, <4 x double>* %b
  ret void
}

define void @ucvtf_v8i32_v8f64(<8 x i32>* %a, <8 x double>* %b) #0 {
; VBITS_EQ_256-LABEL: ucvtf_v8i32_v8f64:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    uunpklo z1.d, z0.s
; VBITS_EQ_256-NEXT:    ext z0.b, z0.b, z0.b, #16
; VBITS_EQ_256-NEXT:    uunpklo z0.d, z0.s
; VBITS_EQ_256-NEXT:    ucvtf z1.d, p0/m, z1.d
; VBITS_EQ_256-NEXT:    ucvtf z0.d, p0/m, z0.d
; VBITS_EQ_256-NEXT:    st1d { z1.d }, p0, [x1]
; VBITS_EQ_256-NEXT:    st1d { z0.d }, p0, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: ucvtf_v8i32_v8f64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl8
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_512-NEXT:    ucvtf z0.d, p0/m, z0.d
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %res = uitofp <8 x i32> %op1 to <8 x double>
  store <8 x double> %res, <8 x double>* %b
  ret void
}

define void @ucvtf_v16i32_v16f64(<16 x i32>* %a, <16 x double>* %b) #0 {
; VBITS_GE_1024-LABEL: ucvtf_v16i32_v16f64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl16
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_1024-NEXT:    ucvtf z0.d, p0/m, z0.d
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %res = uitofp <16 x i32> %op1 to <16 x double>
  store <16 x double> %res, <16 x double>* %b
  ret void
}

define void @ucvtf_v32i32_v32f64(<32 x i32>* %a, <32 x double>* %b) #0 {
; VBITS_GE_2048-LABEL: ucvtf_v32i32_v32f64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl32
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    ucvtf z0.d, p0/m, z0.d
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %res = uitofp <32 x i32> %op1 to <32 x double>
  store <32 x double> %res, <32 x double>* %b
  ret void
}


;
; UCVTF D -> H
;

; Don't use SVE for 64-bit vectors.
define <1 x half> @ucvtf_v1i64_v1f16(<1 x i64> %op1) #0 {
; CHECK-LABEL: ucvtf_v1i64_v1f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $q0
; CHECK-NEXT:    fmov x8, d0
; CHECK-NEXT:    ucvtf h0, x8
; CHECK-NEXT:    ret
  %res = uitofp <1 x i64> %op1 to <1 x half>
  ret <1 x half> %res
}

; v2f16 is not legal for NEON, so use SVE
define <2 x half> @ucvtf_v2i64_v2f16(<2 x i64> %op1) #0 {
; CHECK-LABEL: ucvtf_v2i64_v2f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $q0 killed $q0 def $z0
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    ucvtf z0.h, p0/m, z0.d
; CHECK-NEXT:    uzp1 z0.s, z0.s, z0.s
; CHECK-NEXT:    uzp1 z0.h, z0.h, z0.h
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $z0
; CHECK-NEXT:    ret
  %res = uitofp <2 x i64> %op1 to <2 x half>
  ret <2 x half> %res
}

define <4 x half> @ucvtf_v4i64_v4f16(<4 x i64>* %a) #0 {
; CHECK-LABEL: ucvtf_v4i64_v4f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    ucvtf z0.h, p0/m, z0.d
; CHECK-NEXT:    uzp1 z0.s, z0.s, z0.s
; CHECK-NEXT:    uzp1 z0.h, z0.h, z0.h
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $z0
; CHECK-NEXT:    ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %res = uitofp <4 x i64> %op1 to <4 x half>
  ret <4 x half> %res
}

define <8 x half> @ucvtf_v8i64_v8f16(<8 x i64>* %a) #0 {
; VBITS_EQ_256-LABEL: ucvtf_v8i64_v8f16:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ptrue p0.d
; VBITS_EQ_256-NEXT:    ucvtf z0.h, p0/m, z0.d
; VBITS_EQ_256-NEXT:    ucvtf z1.h, p0/m, z1.d
; VBITS_EQ_256-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_EQ_256-NEXT:    uzp1 z1.s, z1.s, z1.s
; VBITS_EQ_256-NEXT:    uzp1 z2.h, z0.h, z0.h
; VBITS_EQ_256-NEXT:    uzp1 z0.h, z1.h, z1.h
; VBITS_EQ_256-NEXT:    mov v0.d[1], v2.d[0]
; VBITS_EQ_256-NEXT:    // kill: def $q0 killed $q0 killed $z0
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: ucvtf_v8i64_v8f16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.d
; VBITS_GE_512-NEXT:    ucvtf z0.h, p0/m, z0.d
; VBITS_GE_512-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_512-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_512-NEXT:    // kill: def $q0 killed $q0 killed $z0
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %res = uitofp <8 x i64> %op1 to <8 x half>
  ret <8 x half> %res
}

define void @ucvtf_v16i64_v16f16(<16 x i64>* %a, <16 x half>* %b) #0 {
; VBITS_GE_1024-LABEL: ucvtf_v16i64_v16f16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.d
; VBITS_GE_1024-NEXT:    ucvtf z0.h, p0/m, z0.d
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl16
; VBITS_GE_1024-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_1024-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_1024-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %res = uitofp <16 x i64> %op1 to <16 x half>
  store <16 x half> %res, <16 x half>* %b
  ret void
}

define void @ucvtf_v32i64_v32f16(<32 x i64>* %a, <32 x half>* %b) #0 {
; VBITS_GE_2048-LABEL: ucvtf_v32i64_v32f16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.d
; VBITS_GE_2048-NEXT:    ucvtf z0.h, p0/m, z0.d
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl32
; VBITS_GE_2048-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_2048-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_2048-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %res = uitofp <32 x i64> %op1 to <32 x half>
  store <32 x half> %res, <32 x half>* %b
  ret void
}

;
; UCVTF D -> S
;

; Don't use SVE for 64-bit vectors.
define <1 x float> @ucvtf_v1i64_v1f32(<1 x i64> %op1) #0 {
; CHECK-LABEL: ucvtf_v1i64_v1f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $q0
; CHECK-NEXT:    ucvtf v0.2d, v0.2d
; CHECK-NEXT:    fcvtn v0.2s, v0.2d
; CHECK-NEXT:    ret
  %res = uitofp <1 x i64> %op1 to <1 x float>
  ret <1 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x float> @ucvtf_v2i64_v2f32(<2 x i64> %op1) #0 {
; CHECK-LABEL: ucvtf_v2i64_v2f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ucvtf v0.2d, v0.2d
; CHECK-NEXT:    fcvtn v0.2s, v0.2d
; CHECK-NEXT:    ret
  %res = uitofp <2 x i64> %op1 to <2 x float>
  ret <2 x float> %res
}

define <4 x float> @ucvtf_v4i64_v4f32(<4 x i64>* %a) #0 {
; CHECK-LABEL: ucvtf_v4i64_v4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    ucvtf z0.s, p0/m, z0.d
; CHECK-NEXT:    uzp1 z0.s, z0.s, z0.s
; CHECK-NEXT:    // kill: def $q0 killed $q0 killed $z0
; CHECK-NEXT:    ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %res = uitofp <4 x i64> %op1 to <4 x float>
  ret <4 x float> %res
}

define void @ucvtf_v8i64_v8f32(<8 x i64>* %a, <8 x float>* %b) #0 {
; VBITS_EQ_256-LABEL: ucvtf_v8i64_v8f32:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ptrue p0.d
; VBITS_EQ_256-NEXT:    ucvtf z0.s, p0/m, z0.d
; VBITS_EQ_256-NEXT:    ucvtf z1.s, p0/m, z1.d
; VBITS_EQ_256-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_EQ_256-NEXT:    uzp1 z1.s, z1.s, z1.s
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl4
; VBITS_EQ_256-NEXT:    splice z1.s, p0, z1.s, z0.s
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    st1w { z1.s }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: ucvtf_v8i64_v8f32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.d
; VBITS_GE_512-NEXT:    ucvtf z0.s, p0/m, z0.d
; VBITS_GE_512-NEXT:    ptrue p0.s, vl8
; VBITS_GE_512-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %res = uitofp <8 x i64> %op1 to <8 x float>
  store <8 x float> %res, <8 x float>* %b
  ret void
}

define void @ucvtf_v16i64_v16f32(<16 x i64>* %a, <16 x float>* %b) #0 {
; VBITS_GE_1024-LABEL: ucvtf_v16i64_v16f32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.d
; VBITS_GE_1024-NEXT:    ucvtf z0.s, p0/m, z0.d
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl16
; VBITS_GE_1024-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_1024-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %res = uitofp <16 x i64> %op1 to <16 x float>
  store <16 x float> %res, <16 x float>* %b
  ret void
}

define void @ucvtf_v32i64_v32f32(<32 x i64>* %a, <32 x float>* %b) #0 {
; VBITS_GE_2048-LABEL: ucvtf_v32i64_v32f32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.d
; VBITS_GE_2048-NEXT:    ucvtf z0.s, p0/m, z0.d
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl32
; VBITS_GE_2048-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %res = uitofp <32 x i64> %op1 to <32 x float>
  store <32 x float> %res, <32 x float>* %b
  ret void
}

;
; UCVTF D -> D
;

; Don't use SVE for 64-bit vectors.
define <1 x double> @ucvtf_v1i64_v1f64(<1 x i64> %op1) #0 {
; CHECK-LABEL: ucvtf_v1i64_v1f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $q0
; CHECK-NEXT:    fmov x8, d0
; CHECK-NEXT:    ucvtf d0, x8
; CHECK-NEXT:    ret
  %res = uitofp <1 x i64> %op1 to <1 x double>
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @ucvtf_v2i64_v2f64(<2 x i64> %op1) #0 {
; CHECK-LABEL: ucvtf_v2i64_v2f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ucvtf v0.2d, v0.2d
; CHECK-NEXT:    ret
  %res = uitofp <2 x i64> %op1 to <2 x double>
  ret <2 x double> %res
}

define void @ucvtf_v4i64_v4f64(<4 x i64>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: ucvtf_v4i64_v4f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ucvtf z0.d, p0/m, z0.d
; CHECK-NEXT:    st1d { z0.d }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %res = uitofp <4 x i64> %op1 to <4 x double>
  store <4 x double> %res, <4 x double>* %b
  ret void
}

define void @ucvtf_v8i64_v8f64(<8 x i64>* %a, <8 x double>* %b) #0 {
; VBITS_EQ_256-LABEL: ucvtf_v8i64_v8f64:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ucvtf z0.d, p0/m, z0.d
; VBITS_EQ_256-NEXT:    ucvtf z1.d, p0/m, z1.d
; VBITS_EQ_256-NEXT:    st1d { z0.d }, p0, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    st1d { z1.d }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: ucvtf_v8i64_v8f64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ucvtf z0.d, p0/m, z0.d
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %res = uitofp <8 x i64> %op1 to <8 x double>
  store <8 x double> %res, <8 x double>* %b
  ret void
}

define void @ucvtf_v16i64_v16f64(<16 x i64>* %a, <16 x double>* %b) #0 {
; VBITS_GE_1024-LABEL: ucvtf_v16i64_v16f64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ucvtf z0.d, p0/m, z0.d
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %res = uitofp <16 x i64> %op1 to <16 x double>
  store <16 x double> %res, <16 x double>* %b
  ret void
}

define void @ucvtf_v32i64_v32f64(<32 x i64>* %a, <32 x double>* %b) #0 {
; VBITS_GE_2048-LABEL: ucvtf_v32i64_v32f64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ucvtf z0.d, p0/m, z0.d
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %res = uitofp <32 x i64> %op1 to <32 x double>
  store <32 x double> %res, <32 x double>* %b
  ret void
}

;
; SCVTF H -> H
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @scvtf_v4i16_v4f16(<4 x i16> %op1) #0 {
; CHECK-LABEL: scvtf_v4i16_v4f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    scvtf v0.4h, v0.4h
; CHECK-NEXT:    ret
  %res = sitofp <4 x i16> %op1 to <4 x half>
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define void @scvtf_v8i16_v8f16(<8 x i16>* %a, <8 x half>* %b) #0 {
; CHECK-LABEL: scvtf_v8i16_v8f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x0]
; CHECK-NEXT:    scvtf v0.8h, v0.8h
; CHECK-NEXT:    str q0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <8 x i16>, <8 x i16>* %a
  %res = sitofp <8 x i16> %op1 to <8 x half>
  store <8 x half> %res, <8 x half>* %b
  ret void
}

define void @scvtf_v16i16_v16f16(<16 x i16>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: scvtf_v16i16_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    scvtf z0.h, p0/m, z0.h
; CHECK-NEXT:    st1h { z0.h }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %res = sitofp <16 x i16> %op1 to <16 x half>
  store <16 x half> %res, <16 x half>* %b
  ret void
}

define void @scvtf_v32i16_v32f16(<32 x i16>* %a, <32 x half>* %b) #0 {
; VBITS_EQ_256-LABEL: scvtf_v32i16_v32f16:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #16
; VBITS_EQ_256-NEXT:    ptrue p0.h, vl16
; VBITS_EQ_256-NEXT:    ld1h { z0.h }, p0/z, [x0, x8, lsl #1]
; VBITS_EQ_256-NEXT:    ld1h { z1.h }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    scvtf z0.h, p0/m, z0.h
; VBITS_EQ_256-NEXT:    scvtf z1.h, p0/m, z1.h
; VBITS_EQ_256-NEXT:    st1h { z0.h }, p0, [x1, x8, lsl #1]
; VBITS_EQ_256-NEXT:    st1h { z1.h }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: scvtf_v32i16_v32f16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl32
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    scvtf z0.h, p0/m, z0.h
; VBITS_GE_512-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %res = sitofp <32 x i16> %op1 to <32 x half>
  store <32 x half> %res, <32 x half>* %b
  ret void
}

define void @scvtf_v64i16_v64f16(<64 x i16>* %a, <64 x half>* %b) #0 {
; VBITS_GE_1024-LABEL: scvtf_v64i16_v64f16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl64
; VBITS_GE_1024-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    scvtf z0.h, p0/m, z0.h
; VBITS_GE_1024-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %res = sitofp <64 x i16> %op1 to <64 x half>
  store <64 x half> %res, <64 x half>* %b
  ret void
}

define void @scvtf_v128i16_v128f16(<128 x i16>* %a, <128 x half>* %b) #0 {
; VBITS_GE_2048-LABEL: scvtf_v128i16_v128f16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl128
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    scvtf z0.h, p0/m, z0.h
; VBITS_GE_2048-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %res = sitofp <128 x i16> %op1 to <128 x half>
  store <128 x half> %res, <128 x half>* %b
  ret void
}

;
; SCVTF H -> S
;

; Don't use SVE for 64-bit vectors.
define <2 x float> @scvtf_v2i16_v2f32(<2 x i16> %op1) #0 {
; CHECK-LABEL: scvtf_v2i16_v2f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    shl v0.2s, v0.2s, #16
; CHECK-NEXT:    sshr v0.2s, v0.2s, #16
; CHECK-NEXT:    scvtf v0.2s, v0.2s
; CHECK-NEXT:    ret
  %res = sitofp <2 x i16> %op1 to <2 x float>
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @scvtf_v4i16_v4f32(<4 x i16> %op1) #0 {
; CHECK-LABEL: scvtf_v4i16_v4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sshll v0.4s, v0.4h, #0
; CHECK-NEXT:    scvtf v0.4s, v0.4s
; CHECK-NEXT:    ret
  %res = sitofp <4 x i16> %op1 to <4 x float>
  ret <4 x float> %res
}

define void @scvtf_v8i16_v8f32(<8 x i16>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: scvtf_v8i16_v8f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x0]
; CHECK-NEXT:    ptrue p0.s, vl8
; CHECK-NEXT:    sunpklo z0.s, z0.h
; CHECK-NEXT:    scvtf z0.s, p0/m, z0.s
; CHECK-NEXT:    st1w { z0.s }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <8 x i16>, <8 x i16>* %a
  %res = sitofp <8 x i16> %op1 to <8 x float>
  store <8 x float> %res, <8 x float>* %b
  ret void
}

define void @scvtf_v16i16_v16f32(<16 x i16>* %a, <16 x float>* %b) #0 {
; VBITS_EQ_256-LABEL: scvtf_v16i16_v16f32:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    ptrue p0.h, vl16
; VBITS_EQ_256-NEXT:    mov x8, #8
; VBITS_EQ_256-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    sunpklo z1.s, z0.h
; VBITS_EQ_256-NEXT:    ext z0.b, z0.b, z0.b, #16
; VBITS_EQ_256-NEXT:    sunpklo z0.s, z0.h
; VBITS_EQ_256-NEXT:    scvtf z1.s, p0/m, z1.s
; VBITS_EQ_256-NEXT:    scvtf z0.s, p0/m, z0.s
; VBITS_EQ_256-NEXT:    st1w { z1.s }, p0, [x1]
; VBITS_EQ_256-NEXT:    st1w { z0.s }, p0, [x1, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: scvtf_v16i16_v16f32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl16
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    sunpklo z0.s, z0.h
; VBITS_GE_512-NEXT:    scvtf z0.s, p0/m, z0.s
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %res = sitofp <16 x i16> %op1 to <16 x float>
  store <16 x float> %res, <16 x float>* %b
  ret void
}

define void @scvtf_v32i16_v32f32(<32 x i16>* %a, <32 x float>* %b) #0 {
; VBITS_GE_1024-LABEL: scvtf_v32i16_v32f32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl32
; VBITS_GE_1024-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    sunpklo z0.s, z0.h
; VBITS_GE_1024-NEXT:    scvtf z0.s, p0/m, z0.s
; VBITS_GE_1024-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %res = sitofp <32 x i16> %op1 to <32 x float>
  store <32 x float> %res, <32 x float>* %b
  ret void
}

define void @scvtf_v64i16_v64f32(<64 x i16>* %a, <64 x float>* %b) #0 {
; VBITS_GE_2048-LABEL: scvtf_v64i16_v64f32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl64
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    sunpklo z0.s, z0.h
; VBITS_GE_2048-NEXT:    scvtf z0.s, p0/m, z0.s
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %res = sitofp <64 x i16> %op1 to <64 x float>
  store <64 x float> %res, <64 x float>* %b
  ret void
}

;
; SCVTF H -> D
;

; v1i16 is perfered to be widened to v4i16, which pushes the output into SVE types, so use SVE
define <1 x double> @scvtf_v1i16_v1f64(<1 x i16> %op1) #0 {
; CHECK-LABEL: scvtf_v1i16_v1f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $z0
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    sunpklo z0.s, z0.h
; CHECK-NEXT:    sunpklo z0.d, z0.s
; CHECK-NEXT:    scvtf z0.d, p0/m, z0.d
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $z0
; CHECK-NEXT:    ret
  %res = sitofp <1 x i16> %op1 to <1 x double>
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @scvtf_v2i16_v2f64(<2 x i16> %op1) #0 {
; CHECK-LABEL: scvtf_v2i16_v2f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    shl v0.2s, v0.2s, #16
; CHECK-NEXT:    sshr v0.2s, v0.2s, #16
; CHECK-NEXT:    sshll v0.2d, v0.2s, #0
; CHECK-NEXT:    scvtf v0.2d, v0.2d
; CHECK-NEXT:    ret
  %res = sitofp <2 x i16> %op1 to <2 x double>
  ret <2 x double> %res
}

define void @scvtf_v4i16_v4f64(<4 x i16>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: scvtf_v4i16_v4f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr d0, [x0]
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    sunpklo z0.s, z0.h
; CHECK-NEXT:    sunpklo z0.d, z0.s
; CHECK-NEXT:    scvtf z0.d, p0/m, z0.d
; CHECK-NEXT:    st1d { z0.d }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <4 x i16>, <4 x i16>* %a
  %res = sitofp <4 x i16> %op1 to <4 x double>
  store <4 x double> %res, <4 x double>* %b
  ret void
}

define void @scvtf_v8i16_v8f64(<8 x i16>* %a, <8 x double>* %b) #0 {
; VBITS_EQ_256-LABEL: scvtf_v8i16_v8f64:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    ldr q0, [x0]
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ext v1.16b, v0.16b, v0.16b, #8
; VBITS_EQ_256-NEXT:    sunpklo z0.s, z0.h
; VBITS_EQ_256-NEXT:    sunpklo z0.d, z0.s
; VBITS_EQ_256-NEXT:    scvtf z0.d, p0/m, z0.d
; VBITS_EQ_256-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_EQ_256-NEXT:    sunpklo z1.s, z1.h
; VBITS_EQ_256-NEXT:    sunpklo z1.d, z1.s
; VBITS_EQ_256-NEXT:    scvtf z1.d, p0/m, z1.d
; VBITS_EQ_256-NEXT:    st1d { z1.d }, p0, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: scvtf_v8i16_v8f64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ldr q0, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    sunpklo z0.s, z0.h
; VBITS_GE_512-NEXT:    sunpklo z0.d, z0.s
; VBITS_GE_512-NEXT:    scvtf z0.d, p0/m, z0.d
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x i16>, <8 x i16>* %a
  %res = sitofp <8 x i16> %op1 to <8 x double>
  store <8 x double> %res, <8 x double>* %b
  ret void
}

define void @scvtf_v16i16_v16f64(<16 x i16>* %a, <16 x double>* %b) #0 {
; VBITS_GE_1024-LABEL: scvtf_v16i16_v16f64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl16
; VBITS_GE_1024-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    sunpklo z0.s, z0.h
; VBITS_GE_1024-NEXT:    sunpklo z0.d, z0.s
; VBITS_GE_1024-NEXT:    scvtf z0.d, p0/m, z0.d
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %res = sitofp <16 x i16> %op1 to <16 x double>
  store <16 x double> %res, <16 x double>* %b
  ret void
}

define void @scvtf_v32i16_v32f64(<32 x i16>* %a, <32 x double>* %b) #0 {
; VBITS_GE_2048-LABEL: scvtf_v32i16_v32f64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl32
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    sunpklo z0.s, z0.h
; VBITS_GE_2048-NEXT:    sunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    scvtf z0.d, p0/m, z0.d
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %res = sitofp <32 x i16> %op1 to <32 x double>
  store <32 x double> %res, <32 x double>* %b
  ret void
}

;
; SCVTF S -> H
;

; Don't use SVE for 64-bit vectors.
define <2 x half> @scvtf_v2i32_v2f16(<2 x i32> %op1) #0 {
; CHECK-LABEL: scvtf_v2i32_v2f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $q0
; CHECK-NEXT:    scvtf v0.4s, v0.4s
; CHECK-NEXT:    fcvtn v0.4h, v0.4s
; CHECK-NEXT:    ret
  %res = sitofp <2 x i32> %op1 to <2 x half>
  ret <2 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x half> @scvtf_v4i32_v4f16(<4 x i32> %op1) #0 {
; CHECK-LABEL: scvtf_v4i32_v4f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    scvtf v0.4s, v0.4s
; CHECK-NEXT:    fcvtn v0.4h, v0.4s
; CHECK-NEXT:    ret
  %res = sitofp <4 x i32> %op1 to <4 x half>
  ret <4 x half> %res
}

define <8 x half> @scvtf_v8i32_v8f16(<8 x i32>* %a) #0 {
; CHECK-LABEL: scvtf_v8i32_v8f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s, vl8
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    scvtf z0.h, p0/m, z0.s
; CHECK-NEXT:    uzp1 z0.h, z0.h, z0.h
; CHECK-NEXT:    // kill: def $q0 killed $q0 killed $z0
; CHECK-NEXT:    ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %res = sitofp <8 x i32> %op1 to <8 x half>
  ret <8 x half> %res
}

define void @scvtf_v16i32_v16f16(<16 x i32>* %a, <16 x half>* %b) #0 {
; VBITS_EQ_256-LABEL: scvtf_v16i32_v16f16:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #8
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    ld1w { z0.s }, p0/z, [x0, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ld1w { z1.s }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ptrue p0.s
; VBITS_EQ_256-NEXT:    scvtf z0.h, p0/m, z0.s
; VBITS_EQ_256-NEXT:    scvtf z1.h, p0/m, z1.s
; VBITS_EQ_256-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_EQ_256-NEXT:    uzp1 z1.h, z1.h, z1.h
; VBITS_EQ_256-NEXT:    ptrue p0.h, vl8
; VBITS_EQ_256-NEXT:    splice z1.h, p0, z1.h, z0.h
; VBITS_EQ_256-NEXT:    ptrue p0.h, vl16
; VBITS_EQ_256-NEXT:    st1h { z1.h }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: scvtf_v16i32_v16f16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.s
; VBITS_GE_512-NEXT:    scvtf z0.h, p0/m, z0.s
; VBITS_GE_512-NEXT:    ptrue p0.h, vl16
; VBITS_GE_512-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_512-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %res = sitofp <16 x i32> %op1 to <16 x half>
  store <16 x half> %res, <16 x half>* %b
  ret void
}

define void @scvtf_v32i32_v32f16(<32 x i32>* %a, <32 x half>* %b) #0 {
; VBITS_GE_1024-LABEL: scvtf_v32i32_v32f16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.s
; VBITS_GE_1024-NEXT:    scvtf z0.h, p0/m, z0.s
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl32
; VBITS_GE_1024-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_1024-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %res = sitofp <32 x i32> %op1 to <32 x half>
  store <32 x half> %res, <32 x half>* %b
  ret void
}

define void @scvtf_v64i32_v64f16(<64 x i32>* %a, <64 x half>* %b) #0 {
; VBITS_GE_2048-LABEL: scvtf_v64i32_v64f16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.s
; VBITS_GE_2048-NEXT:    scvtf z0.h, p0/m, z0.s
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl64
; VBITS_GE_2048-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_2048-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %res = sitofp <64 x i32> %op1 to <64 x half>
  store <64 x half> %res, <64 x half>* %b
  ret void
}

;
; SCVTF S -> S
;

; Don't use SVE for 64-bit vectors.
define <2 x float> @scvtf_v2i32_v2f32(<2 x i32> %op1) #0 {
; CHECK-LABEL: scvtf_v2i32_v2f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    scvtf v0.2s, v0.2s
; CHECK-NEXT:    ret
  %res = sitofp <2 x i32> %op1 to <2 x float>
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @scvtf_v4i32_v4f32(<4 x i32> %op1) #0 {
; CHECK-LABEL: scvtf_v4i32_v4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    scvtf v0.4s, v0.4s
; CHECK-NEXT:    ret
  %res = sitofp <4 x i32> %op1 to <4 x float>
  ret <4 x float> %res
}

define void @scvtf_v8i32_v8f32(<8 x i32>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: scvtf_v8i32_v8f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s, vl8
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    scvtf z0.s, p0/m, z0.s
; CHECK-NEXT:    st1w { z0.s }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %res = sitofp <8 x i32> %op1 to <8 x float>
  store <8 x float> %res, <8 x float>* %b
  ret void
}

define void @scvtf_v16i32_v16f32(<16 x i32>* %a, <16 x float>* %b) #0 {
; VBITS_EQ_256-LABEL: scvtf_v16i32_v16f32:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #8
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    ld1w { z0.s }, p0/z, [x0, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ld1w { z1.s }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    scvtf z0.s, p0/m, z0.s
; VBITS_EQ_256-NEXT:    scvtf z1.s, p0/m, z1.s
; VBITS_EQ_256-NEXT:    st1w { z0.s }, p0, [x1, x8, lsl #2]
; VBITS_EQ_256-NEXT:    st1w { z1.s }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: scvtf_v16i32_v16f32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    scvtf z0.s, p0/m, z0.s
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %res = sitofp <16 x i32> %op1 to <16 x float>
  store <16 x float> %res, <16 x float>* %b
  ret void
}

define void @scvtf_v32i32_v32f32(<32 x i32>* %a, <32 x float>* %b) #0 {
; VBITS_GE_1024-LABEL: scvtf_v32i32_v32f32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    scvtf z0.s, p0/m, z0.s
; VBITS_GE_1024-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %res = sitofp <32 x i32> %op1 to <32 x float>
  store <32 x float> %res, <32 x float>* %b
  ret void
}

define void @scvtf_v64i32_v64f32(<64 x i32>* %a, <64 x float>* %b) #0 {
; VBITS_GE_2048-LABEL: scvtf_v64i32_v64f32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    scvtf z0.s, p0/m, z0.s
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %res = sitofp <64 x i32> %op1 to <64 x float>
  store <64 x float> %res, <64 x float>* %b
  ret void
}

;
; SCVTF S -> D
;

; Don't use SVE for 64-bit vectors.
define <1 x double> @scvtf_v1i32_v1f64(<1 x i32> %op1) #0 {
; CHECK-LABEL: scvtf_v1i32_v1f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sshll v0.2d, v0.2s, #0
; CHECK-NEXT:    scvtf v0.2d, v0.2d
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $q0
; CHECK-NEXT:    ret
  %res = sitofp <1 x i32> %op1 to <1 x double>
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @scvtf_v2i32_v2f64(<2 x i32> %op1) #0 {
; CHECK-LABEL: scvtf_v2i32_v2f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sshll v0.2d, v0.2s, #0
; CHECK-NEXT:    scvtf v0.2d, v0.2d
; CHECK-NEXT:    ret
  %res = sitofp <2 x i32> %op1 to <2 x double>
  ret <2 x double> %res
}

define void @scvtf_v4i32_v4f64(<4 x i32>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: scvtf_v4i32_v4f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x0]
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    sunpklo z0.d, z0.s
; CHECK-NEXT:    scvtf z0.d, p0/m, z0.d
; CHECK-NEXT:    st1d { z0.d }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <4 x i32>, <4 x i32>* %a
  %res = sitofp <4 x i32> %op1 to <4 x double>
  store <4 x double> %res, <4 x double>* %b
  ret void
}

define void @scvtf_v8i32_v8f64(<8 x i32>* %a, <8 x double>* %b) #0 {
; VBITS_EQ_256-LABEL: scvtf_v8i32_v8f64:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    sunpklo z1.d, z0.s
; VBITS_EQ_256-NEXT:    ext z0.b, z0.b, z0.b, #16
; VBITS_EQ_256-NEXT:    sunpklo z0.d, z0.s
; VBITS_EQ_256-NEXT:    scvtf z1.d, p0/m, z1.d
; VBITS_EQ_256-NEXT:    scvtf z0.d, p0/m, z0.d
; VBITS_EQ_256-NEXT:    st1d { z1.d }, p0, [x1]
; VBITS_EQ_256-NEXT:    st1d { z0.d }, p0, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: scvtf_v8i32_v8f64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl8
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    sunpklo z0.d, z0.s
; VBITS_GE_512-NEXT:    scvtf z0.d, p0/m, z0.d
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %res = sitofp <8 x i32> %op1 to <8 x double>
  store <8 x double> %res, <8 x double>* %b
  ret void
}

define void @scvtf_v16i32_v16f64(<16 x i32>* %a, <16 x double>* %b) #0 {
; VBITS_GE_1024-LABEL: scvtf_v16i32_v16f64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl16
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    sunpklo z0.d, z0.s
; VBITS_GE_1024-NEXT:    scvtf z0.d, p0/m, z0.d
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %res = sitofp <16 x i32> %op1 to <16 x double>
  store <16 x double> %res, <16 x double>* %b
  ret void
}

define void @scvtf_v32i32_v32f64(<32 x i32>* %a, <32 x double>* %b) #0 {
; VBITS_GE_2048-LABEL: scvtf_v32i32_v32f64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl32
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    sunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    scvtf z0.d, p0/m, z0.d
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %res = sitofp <32 x i32> %op1 to <32 x double>
  store <32 x double> %res, <32 x double>* %b
  ret void
}


;
; SCVTF D -> H
;

; Don't use SVE for 64-bit vectors.
define <1 x half> @scvtf_v1i64_v1f16(<1 x i64> %op1) #0 {
; CHECK-LABEL: scvtf_v1i64_v1f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $q0
; CHECK-NEXT:    fmov x8, d0
; CHECK-NEXT:    scvtf h0, x8
; CHECK-NEXT:    ret
  %res = sitofp <1 x i64> %op1 to <1 x half>
  ret <1 x half> %res
}

; v2f16 is not legal for NEON, so use SVE
define <2 x half> @scvtf_v2i64_v2f16(<2 x i64> %op1) #0 {
; CHECK-LABEL: scvtf_v2i64_v2f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $q0 killed $q0 def $z0
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    scvtf z0.h, p0/m, z0.d
; CHECK-NEXT:    uzp1 z0.s, z0.s, z0.s
; CHECK-NEXT:    uzp1 z0.h, z0.h, z0.h
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $z0
; CHECK-NEXT:    ret
  %res = sitofp <2 x i64> %op1 to <2 x half>
  ret <2 x half> %res
}

define <4 x half> @scvtf_v4i64_v4f16(<4 x i64>* %a) #0 {
; CHECK-LABEL: scvtf_v4i64_v4f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    scvtf z0.h, p0/m, z0.d
; CHECK-NEXT:    uzp1 z0.s, z0.s, z0.s
; CHECK-NEXT:    uzp1 z0.h, z0.h, z0.h
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $z0
; CHECK-NEXT:    ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %res = sitofp <4 x i64> %op1 to <4 x half>
  ret <4 x half> %res
}

define <8 x half> @scvtf_v8i64_v8f16(<8 x i64>* %a) #0 {
; VBITS_EQ_256-LABEL: scvtf_v8i64_v8f16:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ptrue p0.d
; VBITS_EQ_256-NEXT:    scvtf z0.h, p0/m, z0.d
; VBITS_EQ_256-NEXT:    scvtf z1.h, p0/m, z1.d
; VBITS_EQ_256-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_EQ_256-NEXT:    uzp1 z1.s, z1.s, z1.s
; VBITS_EQ_256-NEXT:    uzp1 z2.h, z0.h, z0.h
; VBITS_EQ_256-NEXT:    uzp1 z0.h, z1.h, z1.h
; VBITS_EQ_256-NEXT:    mov v0.d[1], v2.d[0]
; VBITS_EQ_256-NEXT:    // kill: def $q0 killed $q0 killed $z0
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: scvtf_v8i64_v8f16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.d
; VBITS_GE_512-NEXT:    scvtf z0.h, p0/m, z0.d
; VBITS_GE_512-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_512-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_512-NEXT:    // kill: def $q0 killed $q0 killed $z0
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %res = sitofp <8 x i64> %op1 to <8 x half>
  ret <8 x half> %res
}

define void @scvtf_v16i64_v16f16(<16 x i64>* %a, <16 x half>* %b) #0 {
; VBITS_GE_1024-LABEL: scvtf_v16i64_v16f16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.d
; VBITS_GE_1024-NEXT:    scvtf z0.h, p0/m, z0.d
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl16
; VBITS_GE_1024-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_1024-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_1024-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %res = sitofp <16 x i64> %op1 to <16 x half>
  store <16 x half> %res, <16 x half>* %b
  ret void
}

define void @scvtf_v32i64_v32f16(<32 x i64>* %a, <32 x half>* %b) #0 {
; VBITS_GE_2048-LABEL: scvtf_v32i64_v32f16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.d
; VBITS_GE_2048-NEXT:    scvtf z0.h, p0/m, z0.d
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl32
; VBITS_GE_2048-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_2048-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_2048-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %res = sitofp <32 x i64> %op1 to <32 x half>
  store <32 x half> %res, <32 x half>* %b
  ret void
}

;
; SCVTF D -> S
;

; Don't use SVE for 64-bit vectors.
define <1 x float> @scvtf_v1i64_v1f32(<1 x i64> %op1) #0 {
; CHECK-LABEL: scvtf_v1i64_v1f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $q0
; CHECK-NEXT:    scvtf v0.2d, v0.2d
; CHECK-NEXT:    fcvtn v0.2s, v0.2d
; CHECK-NEXT:    ret
  %res = sitofp <1 x i64> %op1 to <1 x float>
  ret <1 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x float> @scvtf_v2i64_v2f32(<2 x i64> %op1) #0 {
; CHECK-LABEL: scvtf_v2i64_v2f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    scvtf v0.2d, v0.2d
; CHECK-NEXT:    fcvtn v0.2s, v0.2d
; CHECK-NEXT:    ret
  %res = sitofp <2 x i64> %op1 to <2 x float>
  ret <2 x float> %res
}

define <4 x float> @scvtf_v4i64_v4f32(<4 x i64>* %a) #0 {
; CHECK-LABEL: scvtf_v4i64_v4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    scvtf z0.s, p0/m, z0.d
; CHECK-NEXT:    uzp1 z0.s, z0.s, z0.s
; CHECK-NEXT:    // kill: def $q0 killed $q0 killed $z0
; CHECK-NEXT:    ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %res = sitofp <4 x i64> %op1 to <4 x float>
  ret <4 x float> %res
}

define void @scvtf_v8i64_v8f32(<8 x i64>* %a, <8 x float>* %b) #0 {
; VBITS_EQ_256-LABEL: scvtf_v8i64_v8f32:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ptrue p0.d
; VBITS_EQ_256-NEXT:    scvtf z0.s, p0/m, z0.d
; VBITS_EQ_256-NEXT:    scvtf z1.s, p0/m, z1.d
; VBITS_EQ_256-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_EQ_256-NEXT:    uzp1 z1.s, z1.s, z1.s
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl4
; VBITS_EQ_256-NEXT:    splice z1.s, p0, z1.s, z0.s
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    st1w { z1.s }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: scvtf_v8i64_v8f32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.d
; VBITS_GE_512-NEXT:    scvtf z0.s, p0/m, z0.d
; VBITS_GE_512-NEXT:    ptrue p0.s, vl8
; VBITS_GE_512-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %res = sitofp <8 x i64> %op1 to <8 x float>
  store <8 x float> %res, <8 x float>* %b
  ret void
}

define void @scvtf_v16i64_v16f32(<16 x i64>* %a, <16 x float>* %b) #0 {
; VBITS_GE_1024-LABEL: scvtf_v16i64_v16f32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.d
; VBITS_GE_1024-NEXT:    scvtf z0.s, p0/m, z0.d
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl16
; VBITS_GE_1024-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_1024-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %res = sitofp <16 x i64> %op1 to <16 x float>
  store <16 x float> %res, <16 x float>* %b
  ret void
}

define void @scvtf_v32i64_v32f32(<32 x i64>* %a, <32 x float>* %b) #0 {
; VBITS_GE_2048-LABEL: scvtf_v32i64_v32f32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.d
; VBITS_GE_2048-NEXT:    scvtf z0.s, p0/m, z0.d
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl32
; VBITS_GE_2048-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %res = sitofp <32 x i64> %op1 to <32 x float>
  store <32 x float> %res, <32 x float>* %b
  ret void
}

;
; SCVTF D -> D
;

; Don't use SVE for 64-bit vectors.
define <1 x double> @scvtf_v1i64_v1f64(<1 x i64> %op1) #0 {
; CHECK-LABEL: scvtf_v1i64_v1f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $q0
; CHECK-NEXT:    fmov x8, d0
; CHECK-NEXT:    scvtf d0, x8
; CHECK-NEXT:    ret
  %res = sitofp <1 x i64> %op1 to <1 x double>
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @scvtf_v2i64_v2f64(<2 x i64> %op1) #0 {
; CHECK-LABEL: scvtf_v2i64_v2f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    scvtf v0.2d, v0.2d
; CHECK-NEXT:    ret
  %res = sitofp <2 x i64> %op1 to <2 x double>
  ret <2 x double> %res
}

define void @scvtf_v4i64_v4f64(<4 x i64>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: scvtf_v4i64_v4f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    scvtf z0.d, p0/m, z0.d
; CHECK-NEXT:    st1d { z0.d }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %res = sitofp <4 x i64> %op1 to <4 x double>
  store <4 x double> %res, <4 x double>* %b
  ret void
}

define void @scvtf_v8i64_v8f64(<8 x i64>* %a, <8 x double>* %b) #0 {
; VBITS_EQ_256-LABEL: scvtf_v8i64_v8f64:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    scvtf z0.d, p0/m, z0.d
; VBITS_EQ_256-NEXT:    scvtf z1.d, p0/m, z1.d
; VBITS_EQ_256-NEXT:    st1d { z0.d }, p0, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    st1d { z1.d }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: scvtf_v8i64_v8f64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    scvtf z0.d, p0/m, z0.d
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %res = sitofp <8 x i64> %op1 to <8 x double>
  store <8 x double> %res, <8 x double>* %b
  ret void
}

define void @scvtf_v16i64_v16f64(<16 x i64>* %a, <16 x double>* %b) #0 {
; VBITS_GE_1024-LABEL: scvtf_v16i64_v16f64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    scvtf z0.d, p0/m, z0.d
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %res = sitofp <16 x i64> %op1 to <16 x double>
  store <16 x double> %res, <16 x double>* %b
  ret void
}

define void @scvtf_v32i64_v32f64(<32 x i64>* %a, <32 x double>* %b) #0 {
; VBITS_GE_2048-LABEL: scvtf_v32i64_v32f64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    scvtf z0.d, p0/m, z0.d
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %res = sitofp <32 x i64> %op1 to <32 x double>
  store <32 x double> %res, <32 x double>* %b
  ret void
}

attributes #0 = { "target-features"="+sve" }

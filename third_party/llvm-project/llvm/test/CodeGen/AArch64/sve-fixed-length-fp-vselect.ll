; RUN: llc -aarch64-sve-vector-bits-min=128  < %s | FileCheck %s -D#VBYTES=16 -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  < %s | FileCheck %s -D#VBYTES=32
; RUN: llc -aarch64-sve-vector-bits-min=384  < %s | FileCheck %s -D#VBYTES=32
; RUN: llc -aarch64-sve-vector-bits-min=512  < %s | FileCheck %s -D#VBYTES=64 -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  < %s | FileCheck %s -D#VBYTES=64 -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  < %s | FileCheck %s -D#VBYTES=64 -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  < %s | FileCheck %s -D#VBYTES=64 -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 < %s | FileCheck %s -D#VBYTES=256 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_GE_2048

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

; Don't use SVE for 64-bit vectors.
define <4 x half> @select_v4f16(<4 x half> %op1, <4 x half> %op2, <4 x i1> %mask) #0 {
; CHECK-LABEL: select_v4f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    shl v2.4h, v2.4h, #15
; CHECK-NEXT:    cmlt v2.4h, v2.4h, #0
; CHECK-NEXT:    bif v0.8b, v1.8b, v2.8b
; CHECK-NEXT:    ret
  %sel = select <4 x i1> %mask, <4 x half> %op1, <4 x half> %op2
  ret <4 x half> %sel
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @select_v8f16(<8 x half> %op1, <8 x half> %op2, <8 x i1> %mask) #0 {
; CHECK-LABEL: select_v8f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ushll v2.8h, v2.8b, #0
; CHECK-NEXT:    shl v2.8h, v2.8h, #15
; CHECK-NEXT:    cmlt v2.8h, v2.8h, #0
; CHECK-NEXT:    bif v0.16b, v1.16b, v2.16b
; CHECK-NEXT:    ret
  %sel = select <8 x i1> %mask, <8 x half> %op1, <8 x half> %op2
  ret <8 x half> %sel
}

define void @select_v16f16(<16 x half>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: select_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue   p0.h, vl16
; CHECK-NEXT:    ld1h    { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h    { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmeq   p1.h, p0/z, z0.h, z1.h
; CHECK-NEXT:    sel     z0.h, p1, z0.h, z1.h
; CHECK-NEXT:    st1h    { z0.h }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %mask = fcmp oeq <16 x half> %op1, %op2
  %sel = select <16 x i1> %mask, <16 x half> %op1, <16 x half> %op2
  store <16 x half> %sel, <16 x half>* %a
  ret void
}

define void @select_v32f16(<32 x half>* %a, <32 x half>* %b) #0 {
; VBITS_GE_512-LABEL: select_v32f16:
; VBITS_GE_512:         // %bb.0:
; VBITS_GE_512-NEXT:      ptrue   p0.h, vl32
; VBITS_GE_512-NEXT:      ld1h    { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:      ld1h    { z1.h }, p0/z, [x1]
; VBITS_GE_512-NEXT:      fcmeq   p1.h, p0/z, z0.h, z1.h
; VBITS_GE_512-NEXT:      sel     z0.h, p1, z0.h, z1.h
; VBITS_GE_512-NEXT:      st1h    { z0.h }, p0, [x0]
; VBITS_GE_512-NEXT:      ret
  %op1 = load <32 x half>, <32 x half>* %a
  %op2 = load <32 x half>, <32 x half>* %b
  %mask = fcmp oeq <32 x half> %op1, %op2
  %sel = select <32 x i1> %mask, <32 x half> %op1, <32 x half> %op2
  store <32 x half> %sel, <32 x half>* %a
  ret void
}

define void @select_v64f16(<64 x half>* %a, <64 x half>* %b) #0 {
; VBITS_GE_1024-LABEL: select_v64f16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue   p0.h, vl64
; VBITS_GE_1024-NEXT:    ld1h    { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1h    { z1.h }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    fcmeq   p1.h, p0/z, z0.h, z1.h
; VBITS_GE_1024-NEXT:    sel     z0.h, p1, z0.h, z1.h
; VBITS_GE_1024-NEXT:    st1h    { z0.h }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <64 x half>, <64 x half>* %a
  %op2 = load <64 x half>, <64 x half>* %b
  %mask = fcmp oeq <64 x half> %op1, %op2
  %sel = select <64 x i1> %mask, <64 x half> %op1, <64 x half> %op2
  store <64 x half> %sel, <64 x half>* %a
  ret void
}

define void @select_v128f16(<128 x half>* %a, <128 x half>* %b) #0 {
; VBITS_GE_2048-LABEL: select_v128f16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue   p0.h, vl128
; VBITS_GE_2048-NEXT:    ld1h    { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1h    { z1.h }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq   p1.h, p0/z, z0.h, z1.h
; VBITS_GE_2048-NEXT:    sel     z0.h, p1, z0.h, z1.h
; VBITS_GE_2048-NEXT:    st1h    { z0.h }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <128 x half>, <128 x half>* %a
  %op2 = load <128 x half>, <128 x half>* %b
  %mask = fcmp oeq <128 x half> %op1, %op2
  %sel = select <128 x i1> %mask, <128 x half> %op1, <128 x half> %op2
  store <128 x half> %sel, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @select_v2f32(<2 x float> %op1, <2 x float> %op2, <2 x i1> %mask) #0 {
; CHECK-LABEL: select_v2f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    shl v2.2s, v2.2s, #31
; CHECK-NEXT:    cmlt v2.2s, v2.2s, #0
; CHECK-NEXT:    bif v0.8b, v1.8b, v2.8b
; CHECK-NEXT:    ret
  %sel = select <2 x i1> %mask, <2 x float> %op1, <2 x float> %op2
  ret <2 x float> %sel
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @select_v4f32(<4 x float> %op1, <4 x float> %op2, <4 x i1> %mask) #0 {
; CHECK-LABEL: select_v4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ushll v2.4s, v2.4h, #0
; CHECK-NEXT:    shl v2.4s, v2.4s, #31
; CHECK-NEXT:    cmlt v2.4s, v2.4s, #0
; CHECK-NEXT:    bif v0.16b, v1.16b, v2.16b
; CHECK-NEXT:    ret
  %sel = select <4 x i1> %mask, <4 x float> %op1, <4 x float> %op2
  ret <4 x float> %sel
}

define void @select_v8f32(<8 x float>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: select_v8f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue   p0.s, vl8
; CHECK-NEXT:    ld1w    { z0.s }, p0/z, [x0]
; CHECK-NEXT:    ld1w    { z1.s }, p0/z, [x1]
; CHECK-NEXT:    fcmeq   p1.s, p0/z, z0.s, z1.s
; CHECK-NEXT:    sel     z0.s, p1, z0.s, z1.s
; CHECK-NEXT:    st1w    { z0.s }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <8 x float>, <8 x float>* %a
  %op2 = load <8 x float>, <8 x float>* %b
  %mask = fcmp oeq <8 x float> %op1, %op2
  %sel = select <8 x i1> %mask, <8 x float> %op1, <8 x float> %op2
  store <8 x float> %sel, <8 x float>* %a
  ret void
}

define void @select_v16f32(<16 x float>* %a, <16 x float>* %b) #0 {
; VBITS_GE_512-LABEL: select_v16f32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512:         ptrue   p0.s, vl16
; VBITS_GE_512:         ld1w    { z0.s }, p0/z, [x0]
; VBITS_GE_512:         ld1w    { z1.s }, p0/z, [x1]
; VBITS_GE_512:         fcmeq   p1.s, p0/z, z0.s, z1.s
; VBITS_GE_512:         sel     z0.s, p1, z0.s, z1.s
; VBITS_GE_512:         st1w    { z0.s }, p0, [x0]
; VBITS_GE_512:         ret
  %op1 = load <16 x float>, <16 x float>* %a
  %op2 = load <16 x float>, <16 x float>* %b
  %mask = fcmp oeq <16 x float> %op1, %op2
  %sel = select <16 x i1> %mask, <16 x float> %op1, <16 x float> %op2
  store <16 x float> %sel, <16 x float>* %a
  ret void
}

define void @select_v32f32(<32 x float>* %a, <32 x float>* %b) #0 {
; VBITS_GE_1024-LABEL: select_v32f32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT     ptrue   p0.s, vl32
; VBITS_GE_1024-NEXT     ld1w    { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT     ld1w    { z1.s }, p0/z, [x1]
; VBITS_GE_1024-NEXT     fcmeq   p1.s, p0/z, z0.s, z1.s
; VBITS_GE_1024-NEXT     sel     z0.s, p1, z0.s, z1.s
; VBITS_GE_1024-NEXT     st1w    { z0.s }, p0, [x0]
; VBITS_GE_1024-NEXT     ret
  %op1 = load <32 x float>, <32 x float>* %a
  %op2 = load <32 x float>, <32 x float>* %b
  %mask = fcmp oeq <32 x float> %op1, %op2
  %sel = select <32 x i1> %mask, <32 x float> %op1, <32 x float> %op2
  store <32 x float> %sel, <32 x float>* %a
  ret void
}

define void @select_v64f32(<64 x float>* %a, <64 x float>* %b) #0 {
; VBITS_GE_2048-LABEL: select_v64f32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue   p0.s, vl64
; VBITS_GE_2048-NEXT:    ld1w    { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1w    { z1.s }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq   p1.s, p0/z, z0.s, z1.s
; VBITS_GE_2048-NEXT:    sel     z0.s, p1, z0.s, z1.s
; VBITS_GE_2048-NEXT:    st1w    { z0.s }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <64 x float>, <64 x float>* %a
  %op2 = load <64 x float>, <64 x float>* %b
  %mask = fcmp oeq <64 x float> %op1, %op2
  %sel = select <64 x i1> %mask, <64 x float> %op1, <64 x float> %op2
  store <64 x float> %sel, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @select_v1f64(<1 x double> %op1, <1 x double> %op2, <1 x i1> %mask) #0 {
; CHECK-LABEL: select_v1f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    tst w0, #0x1
; CHECK-NEXT:    csetm x8, ne
; CHECK-NEXT:    fmov d2, x8
; CHECK-NEXT:    bif v0.8b, v1.8b, v2.8b
; CHECK-NEXT:    ret
  %sel = select <1 x i1> %mask, <1 x double> %op1, <1 x double> %op2
  ret <1 x double> %sel
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @select_v2f64(<2 x double> %op1, <2 x double> %op2, <2 x i1> %mask) #0 {
; CHECK-LABEL: select_v2f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ushll v2.2d, v2.2s, #0
; CHECK-NEXT:    shl v2.2d, v2.2d, #63
; CHECK-NEXT:    cmlt v2.2d, v2.2d, #0
; CHECK-NEXT:    bif v0.16b, v1.16b, v2.16b
; CHECK-NEXT:    ret
  %sel = select <2 x i1> %mask, <2 x double> %op1, <2 x double> %op2
  ret <2 x double> %sel
}

define void @select_v4f64(<4 x double>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: select_v4f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue   p0.d, vl4
; CHECK-NEXT:    ld1d    { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ld1d    { z1.d }, p0/z, [x1]
; CHECK-NEXT:    fcmeq   p1.d, p0/z, z0.d, z1.d
; CHECK-NEXT:    sel     z0.d, p1, z0.d, z1.d
; CHECK-NEXT:    st1d    { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <4 x double>, <4 x double>* %a
  %op2 = load <4 x double>, <4 x double>* %b
  %mask = fcmp oeq <4 x double> %op1, %op2
  %sel = select <4 x i1> %mask, <4 x double> %op1, <4 x double> %op2
  store <4 x double> %sel, <4 x double>* %a
  ret void
}

define void @select_v8f64(<8 x double>* %a, <8 x double>* %b) #0 {
; VBITS_GE_512-LABEL: select_v8f64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue   p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d    { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1d    { z1.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    fcmeq   p1.d, p0/z, z0.d, z1.d
; VBITS_GE_512-NEXT:    sel     z0.d, p1, z0.d, z1.d
; VBITS_GE_512-NEXT:    st1d    { z0.d }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x double>, <8 x double>* %a
  %op2 = load <8 x double>, <8 x double>* %b
  %mask = fcmp oeq <8 x double> %op1, %op2
  %sel = select <8 x i1> %mask, <8 x double> %op1, <8 x double> %op2
  store <8 x double> %sel, <8 x double>* %a
  ret void
}

define void @select_v16f64(<16 x double>* %a, <16 x double>* %b) #0 {
; VBITS_GE_1024-LABEL: select_v16f64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue   p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d    { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1d    { z1.d }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    fcmeq   p1.d, p0/z, z0.d, z1.d
; VBITS_GE_1024-NEXT:    sel     z0.d, p1, z0.d, z1.d
; VBITS_GE_1024-NEXT:    st1d    { z0.d }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x double>, <16 x double>* %a
  %op2 = load <16 x double>, <16 x double>* %b
  %mask = fcmp oeq <16 x double> %op1, %op2
  %sel = select <16 x i1> %mask, <16 x double> %op1, <16 x double> %op2
  store <16 x double> %sel, <16 x double>* %a
  ret void
}

define void @select_v32f64(<32 x double>* %a, <32 x double>* %b) #0 {
; VBITS_GE_2048-LABEL: select_v32f64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue   p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d    { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d    { z1.d }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq   p1.d, p0/z, z0.d, z1.d
; VBITS_GE_2048-NEXT:    sel     z0.d, p1, z0.d, z1.d
; VBITS_GE_2048-NEXT:    st1d    { z0.d }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x double>, <32 x double>* %a
  %op2 = load <32 x double>, <32 x double>* %b
  %mask = fcmp oeq <32 x double> %op1, %op2
  %sel = select <32 x i1> %mask, <32 x double> %op1, <32 x double> %op2
  store <32 x double> %sel, <32 x double>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }

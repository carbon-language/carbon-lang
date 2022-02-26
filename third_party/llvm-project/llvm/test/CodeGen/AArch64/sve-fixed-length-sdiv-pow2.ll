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

define <8 x i8> @sdiv_v8i8(<8 x i8> %op1) #0 {
; CHECK-LABEL: sdiv_v8i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $z0
; CHECK-NEXT:    ptrue p0.b, vl8
; CHECK-NEXT:    asrd z0.b, p0/m, z0.b, #5
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $z0
; CHECK-NEXT:    ret
  %res = sdiv <8 x i8> %op1, shufflevector (<8 x i8> insertelement (<8 x i8> poison, i8 32, i32 0), <8 x i8> poison, <8 x i32> zeroinitializer)
  ret <8 x i8> %res
}

define <16 x i8> @sdiv_v16i8(<16 x i8> %op1) #0 {
; CHECK-LABEL: sdiv_v16i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $q0 killed $q0 def $z0
; CHECK-NEXT:    ptrue p0.b, vl16
; CHECK-NEXT:    asrd z0.b, p0/m, z0.b, #5
; CHECK-NEXT:    // kill: def $q0 killed $q0 killed $z0
; CHECK-NEXT:    ret
  %res = sdiv <16 x i8> %op1, shufflevector (<16 x i8> insertelement (<16 x i8> poison, i8 32, i32 0), <16 x i8> poison, <16 x i32> zeroinitializer)
  ret <16 x i8> %res
}

define void @sdiv_v32i8(<32 x i8>* %a) #0 {
; CHECK-LABEL: sdiv_v32i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.b, vl32
; CHECK-NEXT:    ld1b { z0.b }, p0/z, [x0]
; CHECK-NEXT:    asrd z0.b, p0/m, z0.b, #5
; CHECK-NEXT:    st1b { z0.b }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %res = sdiv <32 x i8> %op1, shufflevector (<32 x i8> insertelement (<32 x i8> poison, i8 32, i32 0), <32 x i8> poison, <32 x i32> zeroinitializer)
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @sdiv_v64i8(<64 x i8>* %a) #0 {
; VBITS_EQ_256-LABEL: sdiv_v64i8:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov w8, #32
; VBITS_EQ_256-NEXT:    ptrue p0.b, vl32
; VBITS_EQ_256-NEXT:    ld1b { z0.b }, p0/z, [x0, x8]
; VBITS_EQ_256-NEXT:    ld1b { z1.b }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    asrd z0.b, p0/m, z0.b, #5
; VBITS_EQ_256-NEXT:    asrd z1.b, p0/m, z1.b, #5
; VBITS_EQ_256-NEXT:    st1b { z0.b }, p0, [x0, x8]
; VBITS_EQ_256-NEXT:    st1b { z1.b }, p0, [x0]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: sdiv_v64i8:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.b, vl64
; VBITS_GE_512-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_512-NEXT:    asrd z0.b, p0/m, z0.b, #5
; VBITS_GE_512-NEXT:    st1b { z0.b }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %res = sdiv <64 x i8> %op1, shufflevector (<64 x i8> insertelement (<64 x i8> poison, i8 32, i32 0), <64 x i8> poison, <64 x i32> zeroinitializer)
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @sdiv_v128i8(<128 x i8>* %a) #0 {
; VBITS_GE_1024-LABEL: sdiv_v128i8:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.b, vl128
; VBITS_GE_1024-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    asrd z0.b, p0/m, z0.b, #5
; VBITS_GE_1024-NEXT:    st1b { z0.b }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %res = sdiv <128 x i8> %op1, shufflevector (<128 x i8> insertelement (<128 x i8> poison, i8 32, i32 0), <128 x i8> poison, <128 x i32> zeroinitializer)
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @sdiv_v256i8(<256 x i8>* %a) #0 {
; VBITS_GE_2048-LABEL: sdiv_v256i8:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.b, vl256
; VBITS_GE_2048-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    asrd z0.b, p0/m, z0.b, #5
; VBITS_GE_2048-NEXT:    st1b { z0.b }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %res = sdiv <256 x i8> %op1, shufflevector (<256 x i8> insertelement (<256 x i8> poison, i8 32, i32 0), <256 x i8> poison, <256 x i32> zeroinitializer)
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

define <4 x i16> @sdiv_v4i16(<4 x i16> %op1) #0 {
; CHECK-LABEL: sdiv_v4i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $z0
; CHECK-NEXT:    ptrue p0.h, vl4
; CHECK-NEXT:    asrd z0.h, p0/m, z0.h, #5
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $z0
; CHECK-NEXT:    ret
  %res = sdiv <4 x i16> %op1, shufflevector (<4 x i16> insertelement (<4 x i16> poison, i16 32, i32 0), <4 x i16> poison, <4 x i32> zeroinitializer)
  ret <4 x i16> %res
}

define <8 x i16> @sdiv_v8i16(<8 x i16> %op1) #0 {
; CHECK-LABEL: sdiv_v8i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $q0 killed $q0 def $z0
; CHECK-NEXT:    ptrue p0.h, vl8
; CHECK-NEXT:    asrd z0.h, p0/m, z0.h, #5
; CHECK-NEXT:    // kill: def $q0 killed $q0 killed $z0
; CHECK-NEXT:    ret
  %res = sdiv <8 x i16> %op1, shufflevector (<8 x i16> insertelement (<8 x i16> poison, i16 32, i32 0), <8 x i16> poison, <8 x i32> zeroinitializer)
  ret <8 x i16> %res
}

define void @sdiv_v16i16(<16 x i16>* %a) #0 {
; CHECK-LABEL: sdiv_v16i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    asrd z0.h, p0/m, z0.h, #5
; CHECK-NEXT:    st1h { z0.h }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %res = sdiv <16 x i16> %op1, shufflevector (<16 x i16> insertelement (<16 x i16> poison, i16 32, i32 0), <16 x i16> poison, <16 x i32> zeroinitializer)
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @sdiv_v32i16(<32 x i16>* %a) #0 {
; VBITS_EQ_256-LABEL: sdiv_v32i16:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #16
; VBITS_EQ_256-NEXT:    ptrue p0.h, vl16
; VBITS_EQ_256-NEXT:    ld1h { z0.h }, p0/z, [x0, x8, lsl #1]
; VBITS_EQ_256-NEXT:    ld1h { z1.h }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    asrd z0.h, p0/m, z0.h, #5
; VBITS_EQ_256-NEXT:    asrd z1.h, p0/m, z1.h, #5
; VBITS_EQ_256-NEXT:    st1h { z0.h }, p0, [x0, x8, lsl #1]
; VBITS_EQ_256-NEXT:    st1h { z1.h }, p0, [x0]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: sdiv_v32i16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl32
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    asrd z0.h, p0/m, z0.h, #5
; VBITS_GE_512-NEXT:    st1h { z0.h }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %res = sdiv <32 x i16> %op1, shufflevector (<32 x i16> insertelement (<32 x i16> poison, i16 32, i32 0), <32 x i16> poison, <32 x i32> zeroinitializer)
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @sdiv_v64i16(<64 x i16>* %a) #0 {
; VBITS_GE_1024-LABEL: sdiv_v64i16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl64
; VBITS_GE_1024-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    asrd z0.h, p0/m, z0.h, #5
; VBITS_GE_1024-NEXT:    st1h { z0.h }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %res = sdiv <64 x i16> %op1, shufflevector (<64 x i16> insertelement (<64 x i16> poison, i16 32, i32 0), <64 x i16> poison, <64 x i32> zeroinitializer)
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @sdiv_v128i16(<128 x i16>* %a) #0 {
; VBITS_GE_2048-LABEL: sdiv_v128i16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl128
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    asrd z0.h, p0/m, z0.h, #5
; VBITS_GE_2048-NEXT:    st1h { z0.h }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %res = sdiv <128 x i16> %op1, shufflevector (<128 x i16> insertelement (<128 x i16> poison, i16 32, i32 0), <128 x i16> poison, <128 x i32> zeroinitializer)
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

define <2 x i32> @sdiv_v2i32(<2 x i32> %op1) #0 {
; CHECK-LABEL: sdiv_v2i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $z0
; CHECK-NEXT:    ptrue p0.s, vl2
; CHECK-NEXT:    asrd z0.s, p0/m, z0.s, #5
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $z0
; CHECK-NEXT:    ret
  %res = sdiv <2 x i32> %op1, shufflevector (<2 x i32> insertelement (<2 x i32> poison, i32 32, i32 0), <2 x i32> poison, <2 x i32> zeroinitializer)
  ret <2 x i32> %res
}

define <4 x i32> @sdiv_v4i32(<4 x i32> %op1) #0 {
; CHECK-LABEL: sdiv_v4i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $q0 killed $q0 def $z0
; CHECK-NEXT:    ptrue p0.s, vl4
; CHECK-NEXT:    asrd z0.s, p0/m, z0.s, #5
; CHECK-NEXT:    // kill: def $q0 killed $q0 killed $z0
; CHECK-NEXT:    ret
  %res = sdiv <4 x i32> %op1, shufflevector (<4 x i32> insertelement (<4 x i32> poison, i32 32, i32 0), <4 x i32> poison, <4 x i32> zeroinitializer)
  ret <4 x i32> %res
}

define void @sdiv_v8i32(<8 x i32>* %a) #0 {
; CHECK-LABEL: sdiv_v8i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s, vl8
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    asrd z0.s, p0/m, z0.s, #5
; CHECK-NEXT:    st1w { z0.s }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %res = sdiv <8 x i32> %op1, shufflevector (<8 x i32> insertelement (<8 x i32> poison, i32 32, i32 0), <8 x i32> poison, <8 x i32> zeroinitializer)
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @sdiv_v16i32(<16 x i32>* %a) #0 {
; VBITS_EQ_256-LABEL: sdiv_v16i32:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #8
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    ld1w { z0.s }, p0/z, [x0, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ld1w { z1.s }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    asrd z0.s, p0/m, z0.s, #5
; VBITS_EQ_256-NEXT:    asrd z1.s, p0/m, z1.s, #5
; VBITS_EQ_256-NEXT:    st1w { z0.s }, p0, [x0, x8, lsl #2]
; VBITS_EQ_256-NEXT:    st1w { z1.s }, p0, [x0]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: sdiv_v16i32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    asrd z0.s, p0/m, z0.s, #5
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %res = sdiv <16 x i32> %op1, shufflevector (<16 x i32> insertelement (<16 x i32> poison, i32 32, i32 0), <16 x i32> poison, <16 x i32> zeroinitializer)
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @sdiv_v32i32(<32 x i32>* %a) #0 {
; VBITS_GE_1024-LABEL: sdiv_v32i32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    asrd z0.s, p0/m, z0.s, #5
; VBITS_GE_1024-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %res = sdiv <32 x i32> %op1, shufflevector (<32 x i32> insertelement (<32 x i32> poison, i32 32, i32 0), <32 x i32> poison, <32 x i32> zeroinitializer)
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @sdiv_v64i32(<64 x i32>* %a) #0 {
; VBITS_GE_2048-LABEL: sdiv_v64i32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    asrd z0.s, p0/m, z0.s, #5
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %res = sdiv <64 x i32> %op1, shufflevector (<64 x i32> insertelement (<64 x i32> poison, i32 32, i32 0), <64 x i32> poison, <64 x i32> zeroinitializer)
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

define <1 x i64> @sdiv_v1i64(<1 x i64> %op1) #0 {
; CHECK-LABEL: sdiv_v1i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $z0
; CHECK-NEXT:    ptrue p0.d, vl1
; CHECK-NEXT:    asrd z0.d, p0/m, z0.d, #5
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $z0
; CHECK-NEXT:    ret
  %res = sdiv <1 x i64> %op1, shufflevector (<1 x i64> insertelement (<1 x i64> poison, i64 32, i32 0), <1 x i64> poison, <1 x i32> zeroinitializer)
  ret <1 x i64> %res
}

; Vector i64 sdiv are not legal for NEON so use SVE when available.
define <2 x i64> @sdiv_v2i64(<2 x i64> %op1) #0 {
; CHECK-LABEL: sdiv_v2i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $q0 killed $q0 def $z0
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    asrd z0.d, p0/m, z0.d, #5
; CHECK-NEXT:    // kill: def $q0 killed $q0 killed $z0
; CHECK-NEXT:    ret
  %res = sdiv <2 x i64> %op1, shufflevector (<2 x i64> insertelement (<2 x i64> poison, i64 32, i32 0), <2 x i64> poison, <2 x i32> zeroinitializer)
  ret <2 x i64> %res
}

define void @sdiv_v4i64(<4 x i64>* %a) #0 {
; CHECK-LABEL: sdiv_v4i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    asrd z0.d, p0/m, z0.d, #5
; CHECK-NEXT:    st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %res = sdiv <4 x i64> %op1, shufflevector (<4 x i64> insertelement (<4 x i64> poison, i64 32, i32 0), <4 x i64> poison, <4 x i32> zeroinitializer)
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @sdiv_v8i64(<8 x i64>* %a) #0 {
; VBITS_EQ_256-LABEL: sdiv_v8i64:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    asrd z0.d, p0/m, z0.d, #5
; VBITS_EQ_256-NEXT:    asrd z1.d, p0/m, z1.d, #5
; VBITS_EQ_256-NEXT:    st1d { z0.d }, p0, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    st1d { z1.d }, p0, [x0]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: sdiv_v8i64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    asrd z0.d, p0/m, z0.d, #5
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %res = sdiv <8 x i64> %op1, shufflevector (<8 x i64> insertelement (<8 x i64> poison, i64 32, i32 0), <8 x i64> poison, <8 x i32> zeroinitializer)
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @sdiv_v16i64(<16 x i64>* %a) #0 {
; VBITS_GE_1024-LABEL: sdiv_v16i64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    asrd z0.d, p0/m, z0.d, #5
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %res = sdiv <16 x i64> %op1, shufflevector (<16 x i64> insertelement (<16 x i64> poison, i64 32, i32 0), <16 x i64> poison, <16 x i32> zeroinitializer)
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @sdiv_v32i64(<32 x i64>* %a) #0 {
; VBITS_GE_2048-LABEL: sdiv_v32i64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    asrd z0.d, p0/m, z0.d, #5
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %res = sdiv <32 x i64> %op1, shufflevector (<32 x i64> insertelement (<32 x i64> poison, i64 32, i32 0), <32 x i64> poison, <32 x i32> zeroinitializer)
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }

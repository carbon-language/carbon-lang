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

; i8

; Don't use SVE for 64-bit vectors.
define <4 x i8> @extract_subvector_v8i8(<8 x i8> %op) #0 {
; CHECK-LABEL: extract_subvector_v8i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    zip2 v0.8b, v0.8b, v0.8b
; CHECK-NEXT:    ret
  %ret = call <4 x i8> @llvm.experimental.vector.extract.v4i8.v8i8(<8 x i8> %op, i64 4)
  ret <4 x i8> %ret
}

; Don't use SVE for 128-bit vectors.
define <8 x i8> @extract_subvector_v16i8(<16 x i8> %op) #0 {
; CHECK-LABEL: extract_subvector_v16i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v0.16b, #8
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $q0
; CHECK-NEXT:    ret
  %ret = call <8 x i8> @llvm.experimental.vector.extract.v8i8.v16i8(<16 x i8> %op, i64 8)
  ret <8 x i8> %ret
}

define void @extract_subvector_v32i8(<32 x i8>* %a, <16 x i8>* %b) #0 {
; CHECK-LABEL: extract_subvector_v32i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.b, vl32
; CHECK-NEXT:    ld1b { z0.b }, p0/z, [x0]
; CHECK-NEXT:    ext z0.b, z0.b, z0.b, #16
; CHECK-NEXT:    str q0, [x1]
; CHECK-NEXT:    ret
  %op = load <32 x i8>, <32 x i8>* %a
  %ret = call <16 x i8> @llvm.experimental.vector.extract.v16i8.v32i8(<32 x i8> %op, i64 16)
  store <16 x i8> %ret, <16 x i8>* %b
  ret void
}

define void @extract_subvector_v64i8(<64 x i8>* %a, <32 x i8>* %b) #0 {
; VBITS_EQ_256-LABEL: extract_subvector_v64i8:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov w8, #32
; VBITS_EQ_256-NEXT:    ptrue p0.b, vl32
; VBITS_EQ_256-NEXT:    ld1b { z0.b }, p0/z, [x0, x8]
; VBITS_EQ_256-NEXT:    st1b { z0.b }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: extract_subvector_v64i8:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.b, vl64
; VBITS_GE_512-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.b, vl32
; VBITS_GE_512-NEXT:    ext z0.b, z0.b, z0.b, #32
; VBITS_GE_512-NEXT:    st1b { z0.b }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op = load <64 x i8>, <64 x i8>* %a
  %ret = call <32 x i8> @llvm.experimental.vector.extract.v32i8.v64i8(<64 x i8> %op, i64 32)
  store <32 x i8> %ret, <32 x i8>* %b
  ret void
}

define void @extract_subvector_v128i8(<128 x i8>* %a, <64 x i8>* %b) #0 {
; VBITS_GE_1024-LABEL: extract_subvector_v128i8:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.b, vl128
; VBITS_GE_1024-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.b, vl64
; VBITS_GE_1024-NEXT:    ext z0.b, z0.b, z0.b, #64
; VBITS_GE_1024-NEXT:    st1b { z0.b }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op = load <128 x i8>, <128 x i8>* %a
  %ret = call <64 x i8> @llvm.experimental.vector.extract.v64i8.v128i8(<128 x i8> %op, i64 64)
  store <64 x i8> %ret, <64 x i8>* %b
  ret void
}

define void @extract_subvector_v256i8(<256 x i8>* %a, <128 x i8>* %b) #0 {
; VBITS_GE_2048-LABEL: extract_subvector_v256i8:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.b, vl256
; VBITS_GE_2048-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.b, vl128
; VBITS_GE_2048-NEXT:    ext z0.b, z0.b, z0.b, #128
; VBITS_GE_2048-NEXT:    st1b { z0.b }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op = load <256 x i8>, <256 x i8>* %a
  %ret = call <128 x i8> @llvm.experimental.vector.extract.v128i8.v256i8(<256 x i8> %op, i64 128)
  store <128 x i8> %ret, <128 x i8>* %b
  ret void
}

; i16

; Don't use SVE for 64-bit vectors.
define <2 x i16> @extract_subvector_v4i16(<4 x i16> %op) #0 {
; CHECK-LABEL: extract_subvector_v4i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $q0
; CHECK-NEXT:    umov w8, v0.h[2]
; CHECK-NEXT:    umov w9, v0.h[3]
; CHECK-NEXT:    fmov s0, w8
; CHECK-NEXT:    mov v0.s[1], w9
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $q0
; CHECK-NEXT:    ret
  %ret = call <2 x i16> @llvm.experimental.vector.extract.v2i16.v4i16(<4 x i16> %op, i64 2)
  ret <2 x i16> %ret
}

; Don't use SVE for 128-bit vectors.
define <4 x i16> @extract_subvector_v8i16(<8 x i16> %op) #0 {
; CHECK-LABEL: extract_subvector_v8i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v0.16b, #8
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $q0
; CHECK-NEXT:    ret
  %ret = call <4 x i16> @llvm.experimental.vector.extract.v4i16.v8i16(<8 x i16> %op, i64 4)
  ret <4 x i16> %ret
}

define void @extract_subvector_v16i16(<16 x i16>* %a, <8 x i16>* %b) #0 {
; CHECK-LABEL: extract_subvector_v16i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ext z0.b, z0.b, z0.b, #16
; CHECK-NEXT:    str q0, [x1]
; CHECK-NEXT:    ret
  %op = load <16 x i16>, <16 x i16>* %a
  %ret = call <8 x i16> @llvm.experimental.vector.extract.v8i16.v16i16(<16 x i16> %op, i64 8)
  store <8 x i16> %ret, <8 x i16>* %b
  ret void
}

define void @extract_subvector_v32i16(<32 x i16>* %a, <16 x i16>* %b) #0 {
; VBITS_EQ_256-LABEL: extract_subvector_v32i16:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #16
; VBITS_EQ_256-NEXT:    ptrue p0.h, vl16
; VBITS_EQ_256-NEXT:    ld1h { z0.h }, p0/z, [x0, x8, lsl #1]
; VBITS_EQ_256-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: extract_subvector_v32i16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl32
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.h, vl16
; VBITS_GE_512-NEXT:    ext z0.b, z0.b, z0.b, #32
; VBITS_GE_512-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op = load <32 x i16>, <32 x i16>* %a
  %ret = call <16 x i16> @llvm.experimental.vector.extract.v16i16.v32i16(<32 x i16> %op, i64 16)
  store <16 x i16> %ret, <16 x i16>* %b
  ret void
}

define void @extract_subvector_v64i16(<64 x i16>* %a, <32 x i16>* %b) #0 {
; VBITS_GE_1024-LABEL: extract_subvector_v64i16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl64
; VBITS_GE_1024-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl32
; VBITS_GE_1024-NEXT:    ext z0.b, z0.b, z0.b, #64
; VBITS_GE_1024-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op = load <64 x i16>, <64 x i16>* %a
  %ret = call <32 x i16> @llvm.experimental.vector.extract.v32i16.v64i16(<64 x i16> %op, i64 32)
  store <32 x i16> %ret, <32 x i16>* %b
  ret void
}

define void @extract_subvector_v128i16(<128 x i16>* %a, <64 x i16>* %b) #0 {
; VBITS_GE_2048-LABEL: extract_subvector_v128i16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl128
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl64
; VBITS_GE_2048-NEXT:    ext z0.b, z0.b, z0.b, #128
; VBITS_GE_2048-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op = load <128 x i16>, <128 x i16>* %a
  %ret = call <64 x i16> @llvm.experimental.vector.extract.v64i16.v128i16(<128 x i16> %op, i64 64)
  store <64 x i16> %ret, <64 x i16>* %b
  ret void
}

; i32

; Don't use SVE for 64-bit vectors.
define <1 x i32> @extract_subvector_v2i32(<2 x i32> %op) #0 {
; CHECK-LABEL: extract_subvector_v2i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $q0
; CHECK-NEXT:    dup v0.2s, v0.s[1]
; CHECK-NEXT:    ret
  %ret = call <1 x i32> @llvm.experimental.vector.extract.v1i32.v2i32(<2 x i32> %op, i64 1)
  ret <1 x i32> %ret
}

; Don't use SVE for 128-bit vectors.
define <2 x i32> @extract_subvector_v4i32(<4 x i32> %op) #0 {
; CHECK-LABEL: extract_subvector_v4i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v0.16b, #8
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $q0
; CHECK-NEXT:    ret
  %ret = call <2 x i32> @llvm.experimental.vector.extract.v2i32.v4i32(<4 x i32> %op, i64 2)
  ret <2 x i32> %ret
}

define void @extract_subvector_v8i32(<8 x i32>* %a, <4 x i32>* %b) #0 {
; CHECK-LABEL: extract_subvector_v8i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s, vl8
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    ext z0.b, z0.b, z0.b, #16
; CHECK-NEXT:    str q0, [x1]
; CHECK-NEXT:    ret
  %op = load <8 x i32>, <8 x i32>* %a
  %ret = call <4 x i32> @llvm.experimental.vector.extract.v4i32.v8i32(<8 x i32> %op, i64 4)
  store <4 x i32> %ret, <4 x i32>* %b
  ret void
}

define void @extract_subvector_v16i32(<16 x i32>* %a, <8 x i32>* %b) #0 {
; VBITS_EQ_256-LABEL: extract_subvector_v16i32:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #8
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    ld1w { z0.s }, p0/z, [x0, x8, lsl #2]
; VBITS_EQ_256-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: extract_subvector_v16i32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.s, vl8
; VBITS_GE_512-NEXT:    ext z0.b, z0.b, z0.b, #32
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op = load <16 x i32>, <16 x i32>* %a
  %ret = call <8 x i32> @llvm.experimental.vector.extract.v8i32.v16i32(<16 x i32> %op, i64 8)
  store <8 x i32> %ret, <8 x i32>* %b
  ret void
}

define void @extract_subvector_v32i32(<32 x i32>* %a, <16 x i32>* %b) #0 {
; VBITS_GE_1024-LABEL: extract_subvector_v32i32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl16
; VBITS_GE_1024-NEXT:    ext z0.b, z0.b, z0.b, #64
; VBITS_GE_1024-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op = load <32 x i32>, <32 x i32>* %a
  %ret = call <16 x i32> @llvm.experimental.vector.extract.v16i32.v32i32(<32 x i32> %op, i64 16)
  store <16 x i32> %ret, <16 x i32>* %b
  ret void
}

define void @extract_subvector_v64i32(<64 x i32>* %a, <32 x i32>* %b) #0 {
; VBITS_GE_2048-LABEL: extract_subvector_v64i32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl32
; VBITS_GE_2048-NEXT:    ext z0.b, z0.b, z0.b, #128
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op = load <64 x i32>, <64 x i32>* %a
  %ret = call <32 x i32> @llvm.experimental.vector.extract.v32i32.v64i32(<64 x i32> %op, i64 32)
  store <32 x i32> %ret, <32 x i32>* %b
  ret void
}

; i64

; Don't use SVE for 128-bit vectors.
define <1 x i64> @extract_subvector_v2i64(<2 x i64> %op) #0 {
; CHECK-LABEL: extract_subvector_v2i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v0.16b, #8
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $q0
; CHECK-NEXT:    ret
  %ret = call <1 x i64> @llvm.experimental.vector.extract.v1i64.v2i64(<2 x i64> %op, i64 1)
  ret <1 x i64> %ret
}

define void @extract_subvector_v4i64(<4 x i64>* %a, <2 x i64>* %b) #0 {
; CHECK-LABEL: extract_subvector_v4i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ext z0.b, z0.b, z0.b, #16
; CHECK-NEXT:    str q0, [x1]
; CHECK-NEXT:    ret
  %op = load <4 x i64>, <4 x i64>* %a
  %ret = call <2 x i64> @llvm.experimental.vector.extract.v2i64.v4i64(<4 x i64> %op, i64 2)
  store <2 x i64> %ret, <2 x i64>* %b
  ret void
}

define void @extract_subvector_v8i64(<8 x i64>* %a, <4 x i64>* %b) #0 {
; VBITS_EQ_256-LABEL: extract_subvector_v8i64:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: extract_subvector_v8i64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.d, vl4
; VBITS_GE_512-NEXT:    ext z0.b, z0.b, z0.b, #32
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op = load <8 x i64>, <8 x i64>* %a
  %ret = call <4 x i64> @llvm.experimental.vector.extract.v4i64.v8i64(<8 x i64> %op, i64 4)
  store <4 x i64> %ret, <4 x i64>* %b
  ret void
}

define void @extract_subvector_v16i64(<16 x i64>* %a, <8 x i64>* %b) #0 {
; VBITS_GE_1024-LABEL: extract_subvector_v16i64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl8
; VBITS_GE_1024-NEXT:    ext z0.b, z0.b, z0.b, #64
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op = load <16 x i64>, <16 x i64>* %a
  %ret = call <8 x i64> @llvm.experimental.vector.extract.v8i64.v16i64(<16 x i64> %op, i64 8)
  store <8 x i64> %ret, <8 x i64>* %b
  ret void
}

define void @extract_subvector_v32i64(<32 x i64>* %a, <16 x i64>* %b) #0 {
; VBITS_GE_2048-LABEL: extract_subvector_v32i64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl16
; VBITS_GE_2048-NEXT:    ext z0.b, z0.b, z0.b, #128
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op = load <32 x i64>, <32 x i64>* %a
  %ret = call <16 x i64> @llvm.experimental.vector.extract.v16i64.v32i64(<32 x i64> %op, i64 16)
  store <16 x i64> %ret, <16 x i64>* %b
  ret void
}

; f16

; Don't use SVE for 64-bit vectors.
define <2 x half> @extract_subvector_v4f16(<4 x half> %op) #0 {
; CHECK-LABEL: extract_subvector_v4f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $q0
; CHECK-NEXT:    dup v0.2s, v0.s[1]
; CHECK-NEXT:    ret
  %ret = call <2 x half> @llvm.experimental.vector.extract.v2f16.v4f16(<4 x half> %op, i64 2)
  ret <2 x half> %ret
}

; Don't use SVE for 128-bit vectors.
define <4 x half> @extract_subvector_v8f16(<8 x half> %op) #0 {
; CHECK-LABEL: extract_subvector_v8f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v0.16b, #8
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $q0
; CHECK-NEXT:    ret
  %ret = call <4 x half> @llvm.experimental.vector.extract.v4f16.v8f16(<8 x half> %op, i64 4)
  ret <4 x half> %ret
}

define void @extract_subvector_v16f16(<16 x half>* %a, <8 x half>* %b) #0 {
; CHECK-LABEL: extract_subvector_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ext z0.b, z0.b, z0.b, #16
; CHECK-NEXT:    str q0, [x1]
; CHECK-NEXT:    ret
  %op = load <16 x half>, <16 x half>* %a
  %ret = call <8 x half> @llvm.experimental.vector.extract.v8f16.v16f16(<16 x half> %op, i64 8)
  store <8 x half> %ret, <8 x half>* %b
  ret void
}

define void @extract_subvector_v32f16(<32 x half>* %a, <16 x half>* %b) #0 {
; VBITS_EQ_256-LABEL: extract_subvector_v32f16:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #16
; VBITS_EQ_256-NEXT:    ptrue p0.h, vl16
; VBITS_EQ_256-NEXT:    ld1h { z0.h }, p0/z, [x0, x8, lsl #1]
; VBITS_EQ_256-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: extract_subvector_v32f16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl32
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.h, vl16
; VBITS_GE_512-NEXT:    ext z0.b, z0.b, z0.b, #32
; VBITS_GE_512-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op = load <32 x half>, <32 x half>* %a
  %ret = call <16 x half> @llvm.experimental.vector.extract.v16f16.v32f16(<32 x half> %op, i64 16)
  store <16 x half> %ret, <16 x half>* %b
  ret void
}

define void @extract_subvector_v64f16(<64 x half>* %a, <32 x half>* %b) #0 {
; VBITS_GE_1024-LABEL: extract_subvector_v64f16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl64
; VBITS_GE_1024-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl32
; VBITS_GE_1024-NEXT:    ext z0.b, z0.b, z0.b, #64
; VBITS_GE_1024-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op = load <64 x half>, <64 x half>* %a
  %ret = call <32 x half> @llvm.experimental.vector.extract.v32f16.v64f16(<64 x half> %op, i64 32)
  store <32 x half> %ret, <32 x half>* %b
  ret void
}

define void @extract_subvector_v128f16(<128 x half>* %a, <64 x half>* %b) #0 {
; VBITS_GE_2048-LABEL: extract_subvector_v128f16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl128
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl64
; VBITS_GE_2048-NEXT:    ext z0.b, z0.b, z0.b, #128
; VBITS_GE_2048-NEXT:    st1h { z0.h }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op = load <128 x half>, <128 x half>* %a
  %ret = call <64 x half> @llvm.experimental.vector.extract.v64f16.v128f16(<128 x half> %op, i64 64)
  store <64 x half> %ret, <64 x half>* %b
  ret void
}

; f32

; Don't use SVE for 64-bit vectors.
define <1 x float> @extract_subvector_v2f32(<2 x float> %op) #0 {
; CHECK-LABEL: extract_subvector_v2f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $q0
; CHECK-NEXT:    dup v0.2s, v0.s[1]
; CHECK-NEXT:    ret
  %ret = call <1 x float> @llvm.experimental.vector.extract.v1f32.v2f32(<2 x float> %op, i64 1)
  ret <1 x float> %ret
}

; Don't use SVE for 128-bit vectors.
define <2 x float> @extract_subvector_v4f32(<4 x float> %op) #0 {
; CHECK-LABEL: extract_subvector_v4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v0.16b, #8
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $q0
; CHECK-NEXT:    ret
  %ret = call <2 x float> @llvm.experimental.vector.extract.v2f32.v4f32(<4 x float> %op, i64 2)
  ret <2 x float> %ret
}

define void @extract_subvector_v8f32(<8 x float>* %a, <4 x float>* %b) #0 {
; CHECK-LABEL: extract_subvector_v8f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s, vl8
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    ext z0.b, z0.b, z0.b, #16
; CHECK-NEXT:    str q0, [x1]
; CHECK-NEXT:    ret
  %op = load <8 x float>, <8 x float>* %a
  %ret = call <4 x float> @llvm.experimental.vector.extract.v4f32.v8f32(<8 x float> %op, i64 4)
  store <4 x float> %ret, <4 x float>* %b
  ret void
}

define void @extract_subvector_v16f32(<16 x float>* %a, <8 x float>* %b) #0 {
; VBITS_EQ_256-LABEL: extract_subvector_v16f32:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #8
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    ld1w { z0.s }, p0/z, [x0, x8, lsl #2]
; VBITS_EQ_256-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: extract_subvector_v16f32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.s, vl8
; VBITS_GE_512-NEXT:    ext z0.b, z0.b, z0.b, #32
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op = load <16 x float>, <16 x float>* %a
  %ret = call <8 x float> @llvm.experimental.vector.extract.v8f32.v16f32(<16 x float> %op, i64 8)
  store <8 x float> %ret, <8 x float>* %b
  ret void
}

define void @extract_subvector_v32f32(<32 x float>* %a, <16 x float>* %b) #0 {
; VBITS_GE_1024-LABEL: extract_subvector_v32f32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl16
; VBITS_GE_1024-NEXT:    ext z0.b, z0.b, z0.b, #64
; VBITS_GE_1024-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op = load <32 x float>, <32 x float>* %a
  %ret = call <16 x float> @llvm.experimental.vector.extract.v16f32.v32f32(<32 x float> %op, i64 16)
  store <16 x float> %ret, <16 x float>* %b
  ret void
}

define void @extract_subvector_v64f32(<64 x float>* %a, <32 x float>* %b) #0 {
; VBITS_GE_2048-LABEL: extract_subvector_v64f32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl32
; VBITS_GE_2048-NEXT:    ext z0.b, z0.b, z0.b, #128
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op = load <64 x float>, <64 x float>* %a
  %ret = call <32 x float> @llvm.experimental.vector.extract.v32f32.v64f32(<64 x float> %op, i64 32)
  store <32 x float> %ret, <32 x float>* %b
  ret void
}

; f64

; Don't use SVE for 128-bit vectors.
define <1 x double> @extract_subvector_v2f64(<2 x double> %op) #0 {
; CHECK-LABEL: extract_subvector_v2f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v0.16b, #8
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $q0
; CHECK-NEXT:    ret
  %ret = call <1 x double> @llvm.experimental.vector.extract.v1f64.v2f64(<2 x double> %op, i64 1)
  ret <1 x double> %ret
}

define void @extract_subvector_v4f64(<4 x double>* %a, <2 x double>* %b) #0 {
; CHECK-LABEL: extract_subvector_v4f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ext z0.b, z0.b, z0.b, #16
; CHECK-NEXT:    str q0, [x1]
; CHECK-NEXT:    ret
  %op = load <4 x double>, <4 x double>* %a
  %ret = call <2 x double> @llvm.experimental.vector.extract.v2f64.v4f64(<4 x double> %op, i64 2)
  store <2 x double> %ret, <2 x double>* %b
  ret void
}

define void @extract_subvector_v8f64(<8 x double>* %a, <4 x double>* %b) #0 {
; VBITS_EQ_256-LABEL: extract_subvector_v8f64:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: extract_subvector_v8f64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.d, vl4
; VBITS_GE_512-NEXT:    ext z0.b, z0.b, z0.b, #32
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op = load <8 x double>, <8 x double>* %a
  %ret = call <4 x double> @llvm.experimental.vector.extract.v4f64.v8f64(<8 x double> %op, i64 4)
  store <4 x double> %ret, <4 x double>* %b
  ret void
}

define void @extract_subvector_v16f64(<16 x double>* %a, <8 x double>* %b) #0 {
; VBITS_GE_1024-LABEL: extract_subvector_v16f64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl8
; VBITS_GE_1024-NEXT:    ext z0.b, z0.b, z0.b, #64
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op = load <16 x double>, <16 x double>* %a
  %ret = call <8 x double> @llvm.experimental.vector.extract.v8f64.v16f64(<16 x double> %op, i64 8)
  store <8 x double> %ret, <8 x double>* %b
  ret void
}

define void @extract_subvector_v32f64(<32 x double>* %a, <16 x double>* %b) #0 {
; VBITS_GE_2048-LABEL: extract_subvector_v32f64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl16
; VBITS_GE_2048-NEXT:    ext z0.b, z0.b, z0.b, #128
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op = load <32 x double>, <32 x double>* %a
  %ret = call <16 x double> @llvm.experimental.vector.extract.v16f64.v32f64(<32 x double> %op, i64 16)
  store <16 x double> %ret, <16 x double>* %b
  ret void
}

declare <4 x i8> @llvm.experimental.vector.extract.v4i8.v8i8(<8 x i8>, i64)
declare <8 x i8> @llvm.experimental.vector.extract.v8i8.v16i8(<16 x i8>, i64)
declare <16 x i8> @llvm.experimental.vector.extract.v16i8.v32i8(<32 x i8>, i64)
declare <32 x i8> @llvm.experimental.vector.extract.v32i8.v64i8(<64 x i8>, i64)
declare <64 x i8> @llvm.experimental.vector.extract.v64i8.v128i8(<128 x i8>, i64)
declare <128 x i8> @llvm.experimental.vector.extract.v128i8.v256i8(<256 x i8>, i64)

declare <2 x i16> @llvm.experimental.vector.extract.v2i16.v4i16(<4 x i16>, i64)
declare <4 x i16> @llvm.experimental.vector.extract.v4i16.v8i16(<8 x i16>, i64)
declare <8 x i16> @llvm.experimental.vector.extract.v8i16.v16i16(<16 x i16>, i64)
declare <16 x i16> @llvm.experimental.vector.extract.v16i16.v32i16(<32 x i16>, i64)
declare <32 x i16> @llvm.experimental.vector.extract.v32i16.v64i16(<64 x i16>, i64)
declare <64 x i16> @llvm.experimental.vector.extract.v64i16.v128i16(<128 x i16>, i64)

declare <1 x i32> @llvm.experimental.vector.extract.v1i32.v2i32(<2 x i32>, i64)
declare <2 x i32> @llvm.experimental.vector.extract.v2i32.v4i32(<4 x i32>, i64)
declare <4 x i32> @llvm.experimental.vector.extract.v4i32.v8i32(<8 x i32>, i64)
declare <8 x i32> @llvm.experimental.vector.extract.v8i32.v16i32(<16 x i32>, i64)
declare <16 x i32> @llvm.experimental.vector.extract.v16i32.v32i32(<32 x i32>, i64)
declare <32 x i32> @llvm.experimental.vector.extract.v32i32.v64i32(<64 x i32>, i64)

declare <1 x i64> @llvm.experimental.vector.extract.v1i64.v2i64(<2 x i64>, i64)
declare <2 x i64> @llvm.experimental.vector.extract.v2i64.v4i64(<4 x i64>, i64)
declare <4 x i64> @llvm.experimental.vector.extract.v4i64.v8i64(<8 x i64>, i64)
declare <8 x i64> @llvm.experimental.vector.extract.v8i64.v16i64(<16 x i64>, i64)
declare <16 x i64> @llvm.experimental.vector.extract.v16i64.v32i64(<32 x i64>, i64)

declare <2 x half> @llvm.experimental.vector.extract.v2f16.v4f16(<4 x half>, i64)
declare <4 x half> @llvm.experimental.vector.extract.v4f16.v8f16(<8 x half>, i64)
declare <8 x half> @llvm.experimental.vector.extract.v8f16.v16f16(<16 x half>, i64)
declare <16 x half> @llvm.experimental.vector.extract.v16f16.v32f16(<32 x half>, i64)
declare <32 x half> @llvm.experimental.vector.extract.v32f16.v64f16(<64 x half>, i64)
declare <64 x half> @llvm.experimental.vector.extract.v64f16.v128f16(<128 x half>, i64)

declare <1 x float> @llvm.experimental.vector.extract.v1f32.v2f32(<2 x float>, i64)
declare <2 x float> @llvm.experimental.vector.extract.v2f32.v4f32(<4 x float>, i64)
declare <4 x float> @llvm.experimental.vector.extract.v4f32.v8f32(<8 x float>, i64)
declare <8 x float> @llvm.experimental.vector.extract.v8f32.v16f32(<16 x float>, i64)
declare <16 x float> @llvm.experimental.vector.extract.v16f32.v32f32(<32 x float>, i64)
declare <32 x float> @llvm.experimental.vector.extract.v32f32.v64f32(<64 x float>, i64)

declare <1 x double> @llvm.experimental.vector.extract.v1f64.v2f64(<2 x double>, i64)
declare <2 x double> @llvm.experimental.vector.extract.v2f64.v4f64(<4 x double>, i64)
declare <4 x double> @llvm.experimental.vector.extract.v4f64.v8f64(<8 x double>, i64)
declare <8 x double> @llvm.experimental.vector.extract.v8f64.v16f64(<16 x double>, i64)
declare <16 x double> @llvm.experimental.vector.extract.v16f64.v32f64(<32 x double>, i64)

attributes #0 = { "target-features"="+sve" }

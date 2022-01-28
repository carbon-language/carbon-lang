; RUN: llc -aarch64-sve-vector-bits-min=128  < %s | FileCheck %s -D#VBYTES=16  -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  < %s | FileCheck %s -D#VBYTES=32  -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=384  < %s | FileCheck %s -D#VBYTES=32  -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=512  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
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

;
; Masked Loads
;
define <2 x half> @masked_load_v2f16(<2 x half>* %ap, <2 x half>* %bp) #0 {
; CHECK-LABEL: masked_load_v2f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr s1, [x0]
; CHECK-NEXT:    movi d0, #0000000000000000
; CHECK-NEXT:    ldr s2, [x1]
; CHECK-NEXT:    ptrue p0.h, vl4
; CHECK-NEXT:    fcmeq v1.4h, v1.4h, v2.4h
; CHECK-NEXT:    umov w8, v1.h[0]
; CHECK-NEXT:    umov w9, v1.h[1]
; CHECK-NEXT:    fmov s1, w8
; CHECK-NEXT:    mov v1.s[1], w9
; CHECK-NEXT:    shl v1.2s, v1.2s, #16
; CHECK-NEXT:    sshr v1.2s, v1.2s, #16
; CHECK-NEXT:    fmov w8, s1
; CHECK-NEXT:    mov w9, v1.s[1]
; CHECK-NEXT:    mov v0.h[0], w8
; CHECK-NEXT:    mov v0.h[1], w9
; CHECK-NEXT:    shl v0.4h, v0.4h, #15
; CHECK-NEXT:    cmlt v0.4h, v0.4h, #0
; CHECK-NEXT:    cmpne p0.h, p0/z, z0.h, #0
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $z0
; CHECK-NEXT:    ret
  %a = load <2 x half>, <2 x half>* %ap
  %b = load <2 x half>, <2 x half>* %bp
  %mask = fcmp oeq <2 x half> %a, %b
  %load = call <2 x half> @llvm.masked.load.v2f16(<2 x half>* %ap, i32 8, <2 x i1> %mask, <2 x half> zeroinitializer)
  ret <2 x half> %load
}

define <2 x float> @masked_load_v2f32(<2 x float>* %ap, <2 x float>* %bp) #0 {
; CHECK-LABEL: masked_load_v2f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr d0, [x0]
; CHECK-NEXT:    ptrue p0.s, vl2
; CHECK-NEXT:    ldr d1, [x1]
; CHECK-NEXT:    fcmeq v0.2s, v0.2s, v1.2s
; CHECK-NEXT:    cmpne p0.s, p0/z, z0.s, #0
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $z0
; CHECK-NEXT:    ret
  %a = load <2 x float>, <2 x float>* %ap
  %b = load <2 x float>, <2 x float>* %bp
  %mask = fcmp oeq <2 x float> %a, %b
  %load = call <2 x float> @llvm.masked.load.v2f32(<2 x float>* %ap, i32 8, <2 x i1> %mask, <2 x float> zeroinitializer)
  ret <2 x float> %load
}

define <4 x float> @masked_load_v4f32(<4 x float>* %ap, <4 x float>* %bp) #0 {
; CHECK-LABEL: masked_load_v4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x0]
; CHECK-NEXT:    ptrue p0.s, vl4
; CHECK-NEXT:    ldr q1, [x1]
; CHECK-NEXT:    fcmeq v0.4s, v0.4s, v1.4s
; CHECK-NEXT:    cmpne p0.s, p0/z, z0.s, #0
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    // kill: def $q0 killed $q0 killed $z0
; CHECK-NEXT:    ret
  %a = load <4 x float>, <4 x float>* %ap
  %b = load <4 x float>, <4 x float>* %bp
  %mask = fcmp oeq <4 x float> %a, %b
  %load = call <4 x float> @llvm.masked.load.v4f32(<4 x float>* %ap, i32 8, <4 x i1> %mask, <4 x float> zeroinitializer)
  ret <4 x float> %load
}

define <8 x float> @masked_load_v8f32(<8 x float>* %ap, <8 x float>* %bp) #0 {
; CHECK-LABEL: masked_load_v8f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s, vl8
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    ld1w { z1.s }, p0/z, [x1]
; CHECK-NEXT:    fcmeq p1.s, p0/z, z0.s, z1.s
; CHECK-NEXT:    ld1w { z0.s }, p1/z, [x0]
; CHECK-NEXT:    st1w { z0.s }, p0, [x8]
; CHECK-NEXT:    ret
  %a = load <8 x float>, <8 x float>* %ap
  %b = load <8 x float>, <8 x float>* %bp
  %mask = fcmp oeq <8 x float> %a, %b
  %load = call <8 x float> @llvm.masked.load.v8f32(<8 x float>* %ap, i32 8, <8 x i1> %mask, <8 x float> zeroinitializer)
  ret <8 x float> %load
}

define <16 x float> @masked_load_v16f32(<16 x float>* %ap, <16 x float>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_v16f32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_512-NEXT:    fcmeq p1.s, p0/z, z0.s, z1.s
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %a = load <16 x float>, <16 x float>* %ap
  %b = load <16 x float>, <16 x float>* %bp
  %mask = fcmp oeq <16 x float> %a, %b
  %load = call <16 x float> @llvm.masked.load.v16f32(<16 x float>* %ap, i32 8, <16 x i1> %mask, <16 x float> zeroinitializer)
  ret <16 x float> %load
}

define <32 x float> @masked_load_v32f32(<32 x float>* %ap, <32 x float>* %bp) #0 {
; VBITS_GE_1024-LABEL: masked_load_v32f32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    fcmeq p1.s, p0/z, z0.s, z1.s
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p1/z, [x0]
; VBITS_GE_1024-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_1024-NEXT:    ret
  %a = load <32 x float>, <32 x float>* %ap
  %b = load <32 x float>, <32 x float>* %bp
  %mask = fcmp oeq <32 x float> %a, %b
  %load = call <32 x float> @llvm.masked.load.v32f32(<32 x float>* %ap, i32 8, <32 x i1> %mask, <32 x float> zeroinitializer)
  ret <32 x float> %load
}

define <64 x float> @masked_load_v64f32(<64 x float>* %ap, <64 x float>* %bp) #0 {
; VBITS_GE_2048-LABEL: masked_load_v64f32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p1.s, p0/z, z0.s, z1.s
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p1/z, [x0]
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_2048-NEXT:    ret
  %a = load <64 x float>, <64 x float>* %ap
  %b = load <64 x float>, <64 x float>* %bp
  %mask = fcmp oeq <64 x float> %a, %b
  %load = call <64 x float> @llvm.masked.load.v64f32(<64 x float>* %ap, i32 8, <64 x i1> %mask, <64 x float> zeroinitializer)
  ret <64 x float> %load
}

define <64 x i8> @masked_load_v64i8(<64 x i8>* %ap, <64 x i8>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_v64i8:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.b, vl64
; VBITS_GE_512-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1b { z1.b }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.b, p0/z, z0.b, z1.b
; VBITS_GE_512-NEXT:    ld1b { z0.b }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1b { z0.b }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %a = load <64 x i8>, <64 x i8>* %ap
  %b = load <64 x i8>, <64 x i8>* %bp
  %mask = icmp eq <64 x i8> %a, %b
  %load = call <64 x i8> @llvm.masked.load.v64i8(<64 x i8>* %ap, i32 8, <64 x i1> %mask, <64 x i8> undef)
  ret <64 x i8> %load
}

define <32 x i16> @masked_load_v32i16(<32 x i16>* %ap, <32 x i16>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_v32i16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl32
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.h, p0/z, z0.h, z1.h
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1h { z0.h }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %a = load <32 x i16>, <32 x i16>* %ap
  %b = load <32 x i16>, <32 x i16>* %bp
  %mask = icmp eq <32 x i16> %a, %b
  %load = call <32 x i16> @llvm.masked.load.v32i16(<32 x i16>* %ap, i32 8, <32 x i1> %mask, <32 x i16> undef)
  ret <32 x i16> %load
}

define <16 x i32> @masked_load_v16i32(<16 x i32>* %ap, <16 x i32>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_v16i32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.s, p0/z, z0.s, z1.s
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %a = load <16 x i32>, <16 x i32>* %ap
  %b = load <16 x i32>, <16 x i32>* %bp
  %mask = icmp eq <16 x i32> %a, %b
  %load = call <16 x i32> @llvm.masked.load.v16i32(<16 x i32>* %ap, i32 8, <16 x i1> %mask, <16 x i32> undef)
  ret <16 x i32> %load
}

define <8 x i64> @masked_load_v8i64(<8 x i64>* %ap, <8 x i64>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_v8i64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.d, p0/z, z0.d, z1.d
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %a = load <8 x i64>, <8 x i64>* %ap
  %b = load <8 x i64>, <8 x i64>* %bp
  %mask = icmp eq <8 x i64> %a, %b
  %load = call <8 x i64> @llvm.masked.load.v8i64(<8 x i64>* %ap, i32 8, <8 x i1> %mask, <8 x i64> undef)
  ret <8 x i64> %load
}

define <8 x i64> @masked_load_passthru_v8i64(<8 x i64>* %ap, <8 x i64>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_passthru_v8i64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.d, p0/z, z0.d, z1.d
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p1/z, [x0]
; VBITS_GE_512-NEXT:    sel z0.d, p1, z0.d, z1.d
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %a = load <8 x i64>, <8 x i64>* %ap
  %b = load <8 x i64>, <8 x i64>* %bp
  %mask = icmp eq <8 x i64> %a, %b
  %load = call <8 x i64> @llvm.masked.load.v8i64(<8 x i64>* %ap, i32 8, <8 x i1> %mask, <8 x i64> %b)
  ret <8 x i64> %load
}

define <8 x double> @masked_load_passthru_v8f64(<8 x double>* %ap, <8 x double>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_passthru_v8f64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    fcmeq p1.d, p0/z, z0.d, z1.d
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p1/z, [x0]
; VBITS_GE_512-NEXT:    sel z0.d, p1, z0.d, z1.d
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %a = load <8 x double>, <8 x double>* %ap
  %b = load <8 x double>, <8 x double>* %bp
  %mask = fcmp oeq <8 x double> %a, %b
  %load = call <8 x double> @llvm.masked.load.v8f64(<8 x double>* %ap, i32 8, <8 x i1> %mask, <8 x double> %b)
  ret <8 x double> %load
}

define <32 x i16> @masked_load_sext_v32i8i16(<32 x i8>* %ap, <32 x i8>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_sext_v32i8i16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.b, vl32
; VBITS_GE_512-NEXT:    ld1b { z0.b }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p0.b, p0/z, z0.b, #0
; VBITS_GE_512-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_512-NEXT:    ld1sb { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.h, vl32
; VBITS_GE_512-NEXT:    st1h { z0.h }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <32 x i8>, <32 x i8>* %bp
  %mask = icmp eq <32 x i8> %b, zeroinitializer
  %load = call <32 x i8> @llvm.masked.load.v32i8(<32 x i8>* %ap, i32 8, <32 x i1> %mask, <32 x i8> undef)
  %ext = sext <32 x i8> %load to <32 x i16>
  ret <32 x i16> %ext
}

define <16 x i32> @masked_load_sext_v16i8i32(<16 x i8>* %ap, <16 x i8>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_sext_v16i8i32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ldr q0, [x1]
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    cmeq v0.16b, v0.16b, #0
; VBITS_GE_512-NEXT:    sunpklo z0.h, z0.b
; VBITS_GE_512-NEXT:    sunpklo z0.s, z0.h
; VBITS_GE_512-NEXT:    cmpne p1.s, p0/z, z0.s, #0
; VBITS_GE_512-NEXT:    ld1sb { z0.s }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <16 x i8>, <16 x i8>* %bp
  %mask = icmp eq <16 x i8> %b, zeroinitializer
  %load = call <16 x i8> @llvm.masked.load.v16i8(<16 x i8>* %ap, i32 8, <16 x i1> %mask, <16 x i8> undef)
  %ext = sext <16 x i8> %load to <16 x i32>
  ret <16 x i32> %ext
}

define <8 x i64> @masked_load_sext_v8i8i64(<8 x i8>* %ap, <8 x i8>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_sext_v8i8i64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ldr d0, [x1]
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    cmeq v0.8b, v0.8b, #0
; VBITS_GE_512-NEXT:    sunpklo z0.h, z0.b
; VBITS_GE_512-NEXT:    sunpklo z0.s, z0.h
; VBITS_GE_512-NEXT:    sunpklo z0.d, z0.s
; VBITS_GE_512-NEXT:    cmpne p1.d, p0/z, z0.d, #0
; VBITS_GE_512-NEXT:    ld1sb { z0.d }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <8 x i8>, <8 x i8>* %bp
  %mask = icmp eq <8 x i8> %b, zeroinitializer
  %load = call <8 x i8> @llvm.masked.load.v8i8(<8 x i8>* %ap, i32 8, <8 x i1> %mask, <8 x i8> undef)
  %ext = sext <8 x i8> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <16 x i32> @masked_load_sext_v16i16i32(<16 x i16>* %ap, <16 x i16>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_sext_v16i16i32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl16
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p0.h, p0/z, z0.h, #0
; VBITS_GE_512-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_512-NEXT:    ld1sh { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <16 x i16>, <16 x i16>* %bp
  %mask = icmp eq <16 x i16> %b, zeroinitializer
  %load = call <16 x i16> @llvm.masked.load.v16i16(<16 x i16>* %ap, i32 8, <16 x i1> %mask, <16 x i16> undef)
  %ext = sext <16 x i16> %load to <16 x i32>
  ret <16 x i32> %ext
}

define <8 x i64> @masked_load_sext_v8i16i64(<8 x i16>* %ap, <8 x i16>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_sext_v8i16i64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ldr q0, [x1]
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    cmeq v0.8h, v0.8h, #0
; VBITS_GE_512-NEXT:    sunpklo z0.s, z0.h
; VBITS_GE_512-NEXT:    sunpklo z0.d, z0.s
; VBITS_GE_512-NEXT:    cmpne p1.d, p0/z, z0.d, #0
; VBITS_GE_512-NEXT:    ld1sh { z0.d }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <8 x i16>, <8 x i16>* %bp
  %mask = icmp eq <8 x i16> %b, zeroinitializer
  %load = call <8 x i16> @llvm.masked.load.v8i16(<8 x i16>* %ap, i32 8, <8 x i1> %mask, <8 x i16> undef)
  %ext = sext <8 x i16> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <8 x i64> @masked_load_sext_v8i32i64(<8 x i32>* %ap, <8 x i32>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_sext_v8i32i64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl8
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p0.s, p0/z, z0.s, #0
; VBITS_GE_512-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_512-NEXT:    ld1sw { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <8 x i32>, <8 x i32>* %bp
  %mask = icmp eq <8 x i32> %b, zeroinitializer
  %load = call <8 x i32> @llvm.masked.load.v8i32(<8 x i32>* %ap, i32 8, <8 x i1> %mask, <8 x i32> undef)
  %ext = sext <8 x i32> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <32 x i16> @masked_load_zext_v32i8i16(<32 x i8>* %ap, <32 x i8>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_zext_v32i8i16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.b, vl32
; VBITS_GE_512-NEXT:    ld1b { z0.b }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p0.b, p0/z, z0.b, #0
; VBITS_GE_512-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_512-NEXT:    ld1b { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.h, vl32
; VBITS_GE_512-NEXT:    st1h { z0.h }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <32 x i8>, <32 x i8>* %bp
  %mask = icmp eq <32 x i8> %b, zeroinitializer
  %load = call <32 x i8> @llvm.masked.load.v32i8(<32 x i8>* %ap, i32 8, <32 x i1> %mask, <32 x i8> undef)
  %ext = zext <32 x i8> %load to <32 x i16>
  ret <32 x i16> %ext
}

define <16 x i32> @masked_load_zext_v16i8i32(<16 x i8>* %ap, <16 x i8>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_zext_v16i8i32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ldr q0, [x1]
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    cmeq v0.16b, v0.16b, #0
; VBITS_GE_512-NEXT:    sunpklo z0.h, z0.b
; VBITS_GE_512-NEXT:    sunpklo z0.s, z0.h
; VBITS_GE_512-NEXT:    cmpne p1.s, p0/z, z0.s, #0
; VBITS_GE_512-NEXT:    ld1b { z0.s }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <16 x i8>, <16 x i8>* %bp
  %mask = icmp eq <16 x i8> %b, zeroinitializer
  %load = call <16 x i8> @llvm.masked.load.v16i8(<16 x i8>* %ap, i32 8, <16 x i1> %mask, <16 x i8> undef)
  %ext = zext <16 x i8> %load to <16 x i32>
  ret <16 x i32> %ext
}

define <8 x i64> @masked_load_zext_v8i8i64(<8 x i8>* %ap, <8 x i8>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_zext_v8i8i64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ldr d0, [x1]
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    cmeq v0.8b, v0.8b, #0
; VBITS_GE_512-NEXT:    sunpklo z0.h, z0.b
; VBITS_GE_512-NEXT:    sunpklo z0.s, z0.h
; VBITS_GE_512-NEXT:    sunpklo z0.d, z0.s
; VBITS_GE_512-NEXT:    cmpne p1.d, p0/z, z0.d, #0
; VBITS_GE_512-NEXT:    ld1b { z0.d }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <8 x i8>, <8 x i8>* %bp
  %mask = icmp eq <8 x i8> %b, zeroinitializer
  %load = call <8 x i8> @llvm.masked.load.v8i8(<8 x i8>* %ap, i32 8, <8 x i1> %mask, <8 x i8> undef)
  %ext = zext <8 x i8> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <16 x i32> @masked_load_zext_v16i16i32(<16 x i16>* %ap, <16 x i16>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_zext_v16i16i32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl16
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p0.h, p0/z, z0.h, #0
; VBITS_GE_512-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_512-NEXT:    ld1h { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <16 x i16>, <16 x i16>* %bp
  %mask = icmp eq <16 x i16> %b, zeroinitializer
  %load = call <16 x i16> @llvm.masked.load.v16i16(<16 x i16>* %ap, i32 8, <16 x i1> %mask, <16 x i16> undef)
  %ext = zext <16 x i16> %load to <16 x i32>
  ret <16 x i32> %ext
}

define <8 x i64> @masked_load_zext_v8i16i64(<8 x i16>* %ap, <8 x i16>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_zext_v8i16i64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ldr q0, [x1]
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    cmeq v0.8h, v0.8h, #0
; VBITS_GE_512-NEXT:    sunpklo z0.s, z0.h
; VBITS_GE_512-NEXT:    sunpklo z0.d, z0.s
; VBITS_GE_512-NEXT:    cmpne p1.d, p0/z, z0.d, #0
; VBITS_GE_512-NEXT:    ld1h { z0.d }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <8 x i16>, <8 x i16>* %bp
  %mask = icmp eq <8 x i16> %b, zeroinitializer
  %load = call <8 x i16> @llvm.masked.load.v8i16(<8 x i16>* %ap, i32 8, <8 x i1> %mask, <8 x i16> undef)
  %ext = zext <8 x i16> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <8 x i64> @masked_load_zext_v8i32i64(<8 x i32>* %ap, <8 x i32>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_zext_v8i32i64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl8
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p0.s, p0/z, z0.s, #0
; VBITS_GE_512-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_512-NEXT:    ld1w { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <8 x i32>, <8 x i32>* %bp
  %mask = icmp eq <8 x i32> %b, zeroinitializer
  %load = call <8 x i32> @llvm.masked.load.v8i32(<8 x i32>* %ap, i32 8, <8 x i1> %mask, <8 x i32> undef)
  %ext = zext <8 x i32> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <32 x i16> @masked_load_sext_v32i8i16_m16(<32 x i8>* %ap, <32 x i16>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_sext_v32i8i16_m16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl32
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.h, p0/z, z0.h, #0
; VBITS_GE_512-NEXT:    ld1sb { z0.h }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1h { z0.h }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <32 x i16>, <32 x i16>* %bp
  %mask = icmp eq <32 x i16> %b, zeroinitializer
  %load = call <32 x i8> @llvm.masked.load.v32i8(<32 x i8>* %ap, i32 8, <32 x i1> %mask, <32 x i8> undef)
  %ext = sext <32 x i8> %load to <32 x i16>
  ret <32 x i16> %ext
}

define <16 x i32> @masked_load_sext_v16i8i32_m32(<16 x i8>* %ap, <16 x i32>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_sext_v16i8i32_m32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.s, p0/z, z0.s, #0
; VBITS_GE_512-NEXT:    ld1sb { z0.s }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <16 x i32>, <16 x i32>* %bp
  %mask = icmp eq <16 x i32> %b, zeroinitializer
  %load = call <16 x i8> @llvm.masked.load.v16i8(<16 x i8>* %ap, i32 8, <16 x i1> %mask, <16 x i8> undef)
  %ext = sext <16 x i8> %load to <16 x i32>
  ret <16 x i32> %ext
}

define <8 x i64> @masked_load_sext_v8i8i64_m64(<8 x i8>* %ap, <8 x i64>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_sext_v8i8i64_m64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.d, p0/z, z0.d, #0
; VBITS_GE_512-NEXT:    ld1sb { z0.d }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <8 x i64>, <8 x i64>* %bp
  %mask = icmp eq <8 x i64> %b, zeroinitializer
  %load = call <8 x i8> @llvm.masked.load.v8i8(<8 x i8>* %ap, i32 8, <8 x i1> %mask, <8 x i8> undef)
  %ext = sext <8 x i8> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <16 x i32> @masked_load_sext_v16i16i32_m32(<16 x i16>* %ap, <16 x i32>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_sext_v16i16i32_m32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.s, p0/z, z0.s, #0
; VBITS_GE_512-NEXT:    ld1sh { z0.s }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <16 x i32>, <16 x i32>* %bp
  %mask = icmp eq <16 x i32> %b, zeroinitializer
  %load = call <16 x i16> @llvm.masked.load.v16i16(<16 x i16>* %ap, i32 8, <16 x i1> %mask, <16 x i16> undef)
  %ext = sext <16 x i16> %load to <16 x i32>
  ret <16 x i32> %ext
}

define <8 x i64> @masked_load_sext_v8i16i64_m64(<8 x i16>* %ap, <8 x i64>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_sext_v8i16i64_m64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.d, p0/z, z0.d, #0
; VBITS_GE_512-NEXT:    ld1sh { z0.d }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <8 x i64>, <8 x i64>* %bp
  %mask = icmp eq <8 x i64> %b, zeroinitializer
  %load = call <8 x i16> @llvm.masked.load.v8i16(<8 x i16>* %ap, i32 8, <8 x i1> %mask, <8 x i16> undef)
  %ext = sext <8 x i16> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <8 x i64> @masked_load_sext_v8i32i64_m64(<8 x i32>* %ap, <8 x i64>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_sext_v8i32i64_m64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.d, p0/z, z0.d, #0
; VBITS_GE_512-NEXT:    ld1sw { z0.d }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <8 x i64>, <8 x i64>* %bp
  %mask = icmp eq <8 x i64> %b, zeroinitializer
  %load = call <8 x i32> @llvm.masked.load.v8i32(<8 x i32>* %ap, i32 8, <8 x i1> %mask, <8 x i32> undef)
  %ext = sext <8 x i32> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <32 x i16> @masked_load_zext_v32i8i16_m16(<32 x i8>* %ap, <32 x i16>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_zext_v32i8i16_m16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl32
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.h, p0/z, z0.h, #0
; VBITS_GE_512-NEXT:    ld1b { z0.h }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1h { z0.h }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <32 x i16>, <32 x i16>* %bp
  %mask = icmp eq <32 x i16> %b, zeroinitializer
  %load = call <32 x i8> @llvm.masked.load.v32i8(<32 x i8>* %ap, i32 8, <32 x i1> %mask, <32 x i8> undef)
  %ext = zext <32 x i8> %load to <32 x i16>
  ret <32 x i16> %ext
}

define <16 x i32> @masked_load_zext_v16i8i32_m32(<16 x i8>* %ap, <16 x i32>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_zext_v16i8i32_m32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.s, p0/z, z0.s, #0
; VBITS_GE_512-NEXT:    ld1b { z0.s }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <16 x i32>, <16 x i32>* %bp
  %mask = icmp eq <16 x i32> %b, zeroinitializer
  %load = call <16 x i8> @llvm.masked.load.v16i8(<16 x i8>* %ap, i32 8, <16 x i1> %mask, <16 x i8> undef)
  %ext = zext <16 x i8> %load to <16 x i32>
  ret <16 x i32> %ext
}

define <8 x i64> @masked_load_zext_v8i8i64_m64(<8 x i8>* %ap, <8 x i64>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_zext_v8i8i64_m64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.d, p0/z, z0.d, #0
; VBITS_GE_512-NEXT:    ld1b { z0.d }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <8 x i64>, <8 x i64>* %bp
  %mask = icmp eq <8 x i64> %b, zeroinitializer
  %load = call <8 x i8> @llvm.masked.load.v8i8(<8 x i8>* %ap, i32 8, <8 x i1> %mask, <8 x i8> undef)
  %ext = zext <8 x i8> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <16 x i32> @masked_load_zext_v16i16i32_m32(<16 x i16>* %ap, <16 x i32>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_zext_v16i16i32_m32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.s, p0/z, z0.s, #0
; VBITS_GE_512-NEXT:    ld1h { z0.s }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <16 x i32>, <16 x i32>* %bp
  %mask = icmp eq <16 x i32> %b, zeroinitializer
  %load = call <16 x i16> @llvm.masked.load.v16i16(<16 x i16>* %ap, i32 8, <16 x i1> %mask, <16 x i16> undef)
  %ext = zext <16 x i16> %load to <16 x i32>
  ret <16 x i32> %ext
}

define <8 x i64> @masked_load_zext_v8i16i64_m64(<8 x i16>* %ap, <8 x i64>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_zext_v8i16i64_m64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.d, p0/z, z0.d, #0
; VBITS_GE_512-NEXT:    ld1h { z0.d }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <8 x i64>, <8 x i64>* %bp
  %mask = icmp eq <8 x i64> %b, zeroinitializer
  %load = call <8 x i16> @llvm.masked.load.v8i16(<8 x i16>* %ap, i32 8, <8 x i1> %mask, <8 x i16> undef)
  %ext = zext <8 x i16> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <8 x i64> @masked_load_zext_v8i32i64_m64(<8 x i32>* %ap, <8 x i64>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_load_zext_v8i32i64_m64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p1.d, p0/z, z0.d, #0
; VBITS_GE_512-NEXT:    ld1w { z0.d }, p1/z, [x0]
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
  %b = load <8 x i64>, <8 x i64>* %bp
  %mask = icmp eq <8 x i64> %b, zeroinitializer
  %load = call <8 x i32> @llvm.masked.load.v8i32(<8 x i32>* %ap, i32 8, <8 x i1> %mask, <8 x i32> undef)
  %ext = zext <8 x i32> %load to <8 x i64>
  ret <8 x i64> %ext
}

define <128 x i16> @masked_load_sext_v128i8i16(<128 x i8>* %ap, <128 x i8>* %bp) #0 {
; VBITS_GE_2048-LABEL: masked_load_sext_v128i8i16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.b, vl128
; VBITS_GE_2048-NEXT:    ld1b { z0.b }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    cmpeq p0.b, p0/z, z0.b, #0
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    ld1sb { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl128
; VBITS_GE_2048-NEXT:    st1h { z0.h }, p0, [x8]
; VBITS_GE_2048-NEXT:    ret
  %b = load <128 x i8>, <128 x i8>* %bp
  %mask = icmp eq <128 x i8> %b, zeroinitializer
  %load = call <128 x i8> @llvm.masked.load.v128i8(<128 x i8>* %ap, i32 8, <128 x i1> %mask, <128 x i8> undef)
  %ext = sext <128 x i8> %load to <128 x i16>
  ret <128 x i16> %ext
}

define <64 x i32> @masked_load_sext_v64i8i32(<64 x i8>* %ap, <64 x i8>* %bp) #0 {
; VBITS_GE_2048-LABEL: masked_load_sext_v64i8i32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.b, vl64
; VBITS_GE_2048-NEXT:    ld1b { z0.b }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    cmpeq p0.b, p0/z, z0.b, #0
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    ld1sb { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_2048-NEXT:    ret
  %b = load <64 x i8>, <64 x i8>* %bp
  %mask = icmp eq <64 x i8> %b, zeroinitializer
  %load = call <64 x i8> @llvm.masked.load.v64i8(<64 x i8>* %ap, i32 8, <64 x i1> %mask, <64 x i8> undef)
  %ext = sext <64 x i8> %load to <64 x i32>
  ret <64 x i32> %ext
}

define <32 x i64> @masked_load_sext_v32i8i64(<32 x i8>* %ap, <32 x i8>* %bp) #0 {
; VBITS_GE_2048-LABEL: masked_load_sext_v32i8i64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.b, vl32
; VBITS_GE_2048-NEXT:    ld1b { z0.b }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    cmpeq p0.b, p0/z, z0.b, #0
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    ld1sb { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_2048-NEXT:    ret
  %b = load <32 x i8>, <32 x i8>* %bp
  %mask = icmp eq <32 x i8> %b, zeroinitializer
  %load = call <32 x i8> @llvm.masked.load.v32i8(<32 x i8>* %ap, i32 8, <32 x i1> %mask, <32 x i8> undef)
  %ext = sext <32 x i8> %load to <32 x i64>
  ret <32 x i64> %ext
}

define <64 x i32> @masked_load_sext_v64i16i32(<64 x i16>* %ap, <64 x i16>* %bp) #0 {
; VBITS_GE_2048-LABEL: masked_load_sext_v64i16i32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl64
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    cmpeq p0.h, p0/z, z0.h, #0
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    ld1sh { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_2048-NEXT:    ret
  %b = load <64 x i16>, <64 x i16>* %bp
  %mask = icmp eq <64 x i16> %b, zeroinitializer
  %load = call <64 x i16> @llvm.masked.load.v64i16(<64 x i16>* %ap, i32 8, <64 x i1> %mask, <64 x i16> undef)
  %ext = sext <64 x i16> %load to <64 x i32>
  ret <64 x i32> %ext
}

define <32 x i64> @masked_load_sext_v32i16i64(<32 x i16>* %ap, <32 x i16>* %bp) #0 {
; VBITS_GE_2048-LABEL: masked_load_sext_v32i16i64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl32
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    cmpeq p0.h, p0/z, z0.h, #0
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    ld1sh { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_2048-NEXT:    ret
  %b = load <32 x i16>, <32 x i16>* %bp
  %mask = icmp eq <32 x i16> %b, zeroinitializer
  %load = call <32 x i16> @llvm.masked.load.v32i16(<32 x i16>* %ap, i32 8, <32 x i1> %mask, <32 x i16> undef)
  %ext = sext <32 x i16> %load to <32 x i64>
  ret <32 x i64> %ext
}

define <32 x i64> @masked_load_sext_v32i32i64(<32 x i32>* %ap, <32 x i32>* %bp) #0 {
; VBITS_GE_2048-LABEL: masked_load_sext_v32i32i64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl32
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    cmpeq p0.s, p0/z, z0.s, #0
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    ld1sw { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_2048-NEXT:    ret
  %b = load <32 x i32>, <32 x i32>* %bp
  %mask = icmp eq <32 x i32> %b, zeroinitializer
  %load = call <32 x i32> @llvm.masked.load.v32i32(<32 x i32>* %ap, i32 8, <32 x i1> %mask, <32 x i32> undef)
  %ext = sext <32 x i32> %load to <32 x i64>
  ret <32 x i64> %ext
}

define <128 x i16> @masked_load_zext_v128i8i16(<128 x i8>* %ap, <128 x i8>* %bp) #0 {
; VBITS_GE_2048-LABEL: masked_load_zext_v128i8i16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.b, vl128
; VBITS_GE_2048-NEXT:    ld1b { z0.b }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    cmpeq p0.b, p0/z, z0.b, #0
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    ld1b { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl128
; VBITS_GE_2048-NEXT:    st1h { z0.h }, p0, [x8]
; VBITS_GE_2048-NEXT:    ret
  %b = load <128 x i8>, <128 x i8>* %bp
  %mask = icmp eq <128 x i8> %b, zeroinitializer
  %load = call <128 x i8> @llvm.masked.load.v128i8(<128 x i8>* %ap, i32 8, <128 x i1> %mask, <128 x i8> undef)
  %ext = zext <128 x i8> %load to <128 x i16>
  ret <128 x i16> %ext
}

define <64 x i32> @masked_load_zext_v64i8i32(<64 x i8>* %ap, <64 x i8>* %bp) #0 {
; VBITS_GE_2048-LABEL: masked_load_zext_v64i8i32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.b, vl64
; VBITS_GE_2048-NEXT:    ld1b { z0.b }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    cmpeq p0.b, p0/z, z0.b, #0
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    ld1b { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_2048-NEXT:    ret
  %b = load <64 x i8>, <64 x i8>* %bp
  %mask = icmp eq <64 x i8> %b, zeroinitializer
  %load = call <64 x i8> @llvm.masked.load.v64i8(<64 x i8>* %ap, i32 8, <64 x i1> %mask, <64 x i8> undef)
  %ext = zext <64 x i8> %load to <64 x i32>
  ret <64 x i32> %ext
}

define <32 x i64> @masked_load_zext_v32i8i64(<32 x i8>* %ap, <32 x i8>* %bp) #0 {
; VBITS_GE_2048-LABEL: masked_load_zext_v32i8i64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.b, vl32
; VBITS_GE_2048-NEXT:    ld1b { z0.b }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    cmpeq p0.b, p0/z, z0.b, #0
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    ld1b { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_2048-NEXT:    ret
  %b = load <32 x i8>, <32 x i8>* %bp
  %mask = icmp eq <32 x i8> %b, zeroinitializer
  %load = call <32 x i8> @llvm.masked.load.v32i8(<32 x i8>* %ap, i32 8, <32 x i1> %mask, <32 x i8> undef)
  %ext = zext <32 x i8> %load to <32 x i64>
  ret <32 x i64> %ext
}

define <64 x i32> @masked_load_zext_v64i16i32(<64 x i16>* %ap, <64 x i16>* %bp) #0 {
; VBITS_GE_2048-LABEL: masked_load_zext_v64i16i32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl64
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    cmpeq p0.h, p0/z, z0.h, #0
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    ld1h { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_2048-NEXT:    ret
  %b = load <64 x i16>, <64 x i16>* %bp
  %mask = icmp eq <64 x i16> %b, zeroinitializer
  %load = call <64 x i16> @llvm.masked.load.v64i16(<64 x i16>* %ap, i32 8, <64 x i1> %mask, <64 x i16> undef)
  %ext = zext <64 x i16> %load to <64 x i32>
  ret <64 x i32> %ext
}

define <32 x i64> @masked_load_zext_v32i16i64(<32 x i16>* %ap, <32 x i16>* %bp) #0 {
; VBITS_GE_2048-LABEL: masked_load_zext_v32i16i64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl32
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    cmpeq p0.h, p0/z, z0.h, #0
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    ld1h { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_2048-NEXT:    ret
  %b = load <32 x i16>, <32 x i16>* %bp
  %mask = icmp eq <32 x i16> %b, zeroinitializer
  %load = call <32 x i16> @llvm.masked.load.v32i16(<32 x i16>* %ap, i32 8, <32 x i1> %mask, <32 x i16> undef)
  %ext = zext <32 x i16> %load to <32 x i64>
  ret <32 x i64> %ext
}

define <32 x i64> @masked_load_zext_v32i32i64(<32 x i32>* %ap, <32 x i32>* %bp) #0 {
; VBITS_GE_2048-LABEL: masked_load_zext_v32i32i64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl32
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    cmpeq p0.s, p0/z, z0.s, #0
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    ld1w { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_2048-NEXT:    ret
  %b = load <32 x i32>, <32 x i32>* %bp
  %mask = icmp eq <32 x i32> %b, zeroinitializer
  %load = call <32 x i32> @llvm.masked.load.v32i32(<32 x i32>* %ap, i32 8, <32 x i1> %mask, <32 x i32> undef)
  %ext = zext <32 x i32> %load to <32 x i64>
  ret <32 x i64> %ext
}

declare <2 x half> @llvm.masked.load.v2f16(<2 x half>*, i32, <2 x i1>, <2 x half>)
declare <2 x float> @llvm.masked.load.v2f32(<2 x float>*, i32, <2 x i1>, <2 x float>)
declare <4 x float> @llvm.masked.load.v4f32(<4 x float>*, i32, <4 x i1>, <4 x float>)
declare <8 x float> @llvm.masked.load.v8f32(<8 x float>*, i32, <8 x i1>, <8 x float>)
declare <16 x float> @llvm.masked.load.v16f32(<16 x float>*, i32, <16 x i1>, <16 x float>)
declare <32 x float> @llvm.masked.load.v32f32(<32 x float>*, i32, <32 x i1>, <32 x float>)
declare <64 x float> @llvm.masked.load.v64f32(<64 x float>*, i32, <64 x i1>, <64 x float>)

declare <128 x i8> @llvm.masked.load.v128i8(<128 x i8>*, i32, <128 x i1>, <128 x i8>)
declare <64 x i8> @llvm.masked.load.v64i8(<64 x i8>*, i32, <64 x i1>, <64 x i8>)
declare <32 x i8> @llvm.masked.load.v32i8(<32 x i8>*, i32, <32 x i1>, <32 x i8>)
declare <16 x i8> @llvm.masked.load.v16i8(<16 x i8>*, i32, <16 x i1>, <16 x i8>)
declare <16 x i16> @llvm.masked.load.v16i16(<16 x i16>*, i32, <16 x i1>, <16 x i16>)
declare <8 x i8> @llvm.masked.load.v8i8(<8 x i8>*, i32, <8 x i1>, <8 x i8>)
declare <8 x i16> @llvm.masked.load.v8i16(<8 x i16>*, i32, <8 x i1>, <8 x i16>)
declare <8 x i32> @llvm.masked.load.v8i32(<8 x i32>*, i32, <8 x i1>, <8 x i32>)
declare <32 x i32> @llvm.masked.load.v32i32(<32 x i32>*, i32, <32 x i1>, <32 x i32>)
declare <32 x i16> @llvm.masked.load.v32i16(<32 x i16>*, i32, <32 x i1>, <32 x i16>)
declare <64 x i16> @llvm.masked.load.v64i16(<64 x i16>*, i32, <64 x i1>, <64 x i16>)
declare <16 x i32> @llvm.masked.load.v16i32(<16 x i32>*, i32, <16 x i1>, <16 x i32>)
declare <8 x i64> @llvm.masked.load.v8i64(<8 x i64>*, i32, <8 x i1>, <8 x i64>)
declare <8 x double> @llvm.masked.load.v8f64(<8 x double>*, i32, <8 x i1>, <8 x double>)

attributes #0 = { "target-features"="+sve" }

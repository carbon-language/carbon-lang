; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -mattr=+neon -o - | FileCheck %s
; RUN: llc < %s -mtriple=aarch64_be-none-linux-gnu -mattr=+neon -o - | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -aarch64-no-strict-align -mattr=+neon -o - | FileCheck %s
; RUN: llc < %s -mtriple=aarch64_be-none-linux-gnu -aarch64-no-strict-align -mattr=+neon -o - | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -aarch64-strict-align -mattr=+neon -o - | FileCheck %s
; RUN: llc < %s -mtriple=aarch64_be-none-linux-gnu -aarch64-strict-align -mattr=+neon -o - | FileCheck %s --check-prefix=BE-STRICT-ALIGN

;; Check element-aligned 128-bit vector load/store - integer
define <16 x i8> @qwordint (<16 x i8>* %head.v16i8,   <8 x i16>* %head.v8i16,   <4 x i32>* %head.v4i32, <2 x i64>* %head.v2i64,
                            <16 x i8>* %tail.v16i8,   <8 x i16>* %tail.v8i16,   <4 x i32>* %tail.v4i32, <2 x i64>* %tail.v2i64) {
; CHECK-LABEL: qwordint
; CHECK: ld1     {v0.16b}, [x0]
; CHECK: ld1     {v1.8h}, [x1]
; CHECK: ld1     {v2.4s}, [x2]
; CHECK: ld1     {v3.2d}, [x3]
; CHECK: st1     {v0.16b}, [x4]
; CHECK: st1     {v1.8h}, [x5]
; CHECK: st1     {v2.4s}, [x6]
; CHECK: st1     {v3.2d}, [x7]
; BE-STRICT-ALIGN-LABEL: qwordint
; BE-STRICT-ALIGN: ldrb
; BE-STRICT-ALIGN: ldrh
; BE-STRICT-ALIGN: ldr
; BE-STRICT-ALIGN: ldr
; BE-STRICT-ALIGN: strb
; BE-STRICT-ALIGN: strh
; BE-STRICT-ALIGN: str
; BE-STRICT-ALIGN: str
entry:
  %val.v16i8 = load <16 x i8>* %head.v16i8, align 1
  %val.v8i16 = load <8 x i16>* %head.v8i16, align 2
  %val.v4i32 = load <4 x i32>* %head.v4i32, align 4
  %val.v2i64 = load <2 x i64>* %head.v2i64, align 8
  store <16 x i8> %val.v16i8, <16 x i8>* %tail.v16i8, align 1
  store <8 x i16> %val.v8i16, <8 x i16>* %tail.v8i16, align 2
  store <4 x i32> %val.v4i32, <4 x i32>* %tail.v4i32, align 4
  store <2 x i64> %val.v2i64, <2 x i64>* %tail.v2i64, align 8
  ret <16 x i8> %val.v16i8
}

;; Check element-aligned 128-bit vector load/store - floating point
define <4 x float> @qwordfloat (<4 x float>* %head.v4f32,   <2 x double>* %head.v2f64,
                                <4 x float>* %tail.v4f32,   <2 x double>* %tail.v2f64) {
; CHECK-LABEL: qwordfloat
; CHECK: ld1     {v0.4s}, [x0]
; CHECK: ld1     {v1.2d}, [x1]
; CHECK: st1     {v0.4s}, [x2]
; CHECK: st1     {v1.2d}, [x3]
; BE-STRICT-ALIGN-LABEL: qwordfloat
; BE-STRICT-ALIGN: ldr
; BE-STRICT-ALIGN: ldr
; BE-STRICT-ALIGN: str
; BE-STRICT-ALIGN: str
entry:
  %val.v4f32 = load <4 x float>*  %head.v4f32, align 4
  %val.v2f64 = load <2 x double>* %head.v2f64, align 8
  store <4 x float>  %val.v4f32, <4 x float>*  %tail.v4f32, align 4
  store <2 x double> %val.v2f64, <2 x double>* %tail.v2f64, align 8
  ret <4 x float> %val.v4f32
}

;; Check element-aligned 64-bit vector load/store - integer
define <8 x i8> @dwordint (<8 x i8>* %head.v8i8,   <4 x i16>* %head.v4i16,   <2 x i32>* %head.v2i32, <1 x i64>* %head.v1i64,
                           <8 x i8>* %tail.v8i8,   <4 x i16>* %tail.v4i16,   <2 x i32>* %tail.v2i32, <1 x i64>* %tail.v1i64) {
; CHECK-LABEL: dwordint
; CHECK: ld1     {v0.8b}, [x0]
; CHECK: ld1     {v1.4h}, [x1]
; CHECK: ld1     {v2.2s}, [x2]
; CHECK: ld1     {v3.1d}, [x3]
; CHECK: st1     {v0.8b}, [x4]
; CHECK: st1     {v1.4h}, [x5]
; CHECK: st1     {v2.2s}, [x6]
; CHECK: st1     {v3.1d}, [x7]
; BE-STRICT-ALIGN-LABEL: dwordint
; BE-STRICT-ALIGN: ldrb
; BE-STRICT-ALIGN: ldrh
; BE-STRICT-ALIGN: ldr
; BE-STRICT-ALIGN: ld1     {v1.1d}, [x3]
; BE-STRICT-ALIGN: strb
; BE-STRICT-ALIGN: strh
; BE-STRICT-ALIGN: str
; BE-STRICT-ALIGN: st1     {v1.1d}, [x7]
entry:
  %val.v8i8  = load <8 x i8>*  %head.v8i8,  align 1
  %val.v4i16 = load <4 x i16>* %head.v4i16, align 2
  %val.v2i32 = load <2 x i32>* %head.v2i32, align 4
  %val.v1i64 = load <1 x i64>* %head.v1i64, align 8
  store <8 x i8>  %val.v8i8,  <8 x i8>*  %tail.v8i8 , align 1
  store <4 x i16> %val.v4i16, <4 x i16>* %tail.v4i16, align 2
  store <2 x i32> %val.v2i32, <2 x i32>* %tail.v2i32, align 4
  store <1 x i64> %val.v1i64, <1 x i64>* %tail.v1i64, align 8
  ret <8 x i8> %val.v8i8
}

;; Check element-aligned 64-bit vector load/store - floating point
define <2 x float> @dwordfloat (<2 x float>* %head.v2f32,   <1 x double>* %head.v1f64,
                                <2 x float>* %tail.v2f32,   <1 x double>* %tail.v1f64) {
; CHECK-LABEL: dwordfloat
; CHECK: ld1     {v0.2s}, [x0]
; CHECK: ld1     {v1.1d}, [x1]
; CHECK: st1     {v0.2s}, [x2]
; CHECK: st1     {v1.1d}, [x3]
; BE-STRICT-ALIGN-LABEL: dwordfloat
; BE-STRICT-ALIGN: ldr
; BE-STRICT-ALIGN: ld1     {v1.1d}, [x1]
; BE-STRICT-ALIGN: str
; BE-STRICT-ALIGN: st1     {v1.1d}, [x3]
entry:
  %val.v2f32 = load <2 x float>*  %head.v2f32, align 4
  %val.v1f64 = load <1 x double>* %head.v1f64, align 8
  store <2 x float>  %val.v2f32, <2 x float>* %tail.v2f32, align 4
  store <1 x double> %val.v1f64, <1 x double>* %tail.v1f64, align 8
  ret <2 x float> %val.v2f32
}

;; Check load/store of 128-bit vectors with less-than 16-byte alignment
define <2 x i64> @align2vi64 (<2 x i64>* %head.byte, <2 x i64>* %head.half, <2 x i64>* %head.word, <2 x i64>* %head.dword,
                              <2 x i64>* %tail.byte, <2 x i64>* %tail.half, <2 x i64>* %tail.word, <2 x i64>* %tail.dword) {
; CHECK-LABEL: align2vi64
; CHECK: ld1     {v0.2d}, [x0]
; CHECK: ld1     {v1.2d}, [x1]
; CHECK: ld1     {v2.2d}, [x2]
; CHECK: ld1     {v3.2d}, [x3]
; CHECK: st1     {v0.2d}, [x4]
; CHECK: st1     {v1.2d}, [x5]
; CHECK: st1     {v2.2d}, [x6]
; CHECK: st1     {v3.2d}, [x7]
; BE-STRICT-ALIGN-LABEL: align2vi64
; BE-STRICT-ALIGN: ldrb     
; BE-STRICT-ALIGN: ldrh     
; BE-STRICT-ALIGN: ldr
; BE-STRICT-ALIGN: strb     
; BE-STRICT-ALIGN: strh     
; BE-STRICT-ALIGN: str
entry:
  %val.byte  = load <2 x i64>* %head.byte,  align 1
  %val.half  = load <2 x i64>* %head.half,  align 2
  %val.word  = load <2 x i64>* %head.word,  align 4
  %val.dword = load <2 x i64>* %head.dword, align 8
  store <2 x i64> %val.byte,  <2 x i64>* %tail.byte,  align 1
  store <2 x i64> %val.half,  <2 x i64>* %tail.half,  align 2
  store <2 x i64> %val.word,  <2 x i64>* %tail.word,  align 4
  store <2 x i64> %val.dword, <2 x i64>* %tail.dword, align 8
  ret <2 x i64> %val.byte
}

;; Check load/store of 64-bit vectors with less-than 8-byte alignment
define <2 x float> @align2vf32 (<2 x float>* %head.byte, <2 x float>* %head.half, <2 x float>* %head.word, <2 x float>* %head.dword,
                                <2 x float>* %tail.byte, <2 x float>* %tail.half, <2 x float>* %tail.word, <2 x float>* %tail.dword) {
; CHECK-LABEL: align2vf32
; CHECK: ld1     {v0.2s}, [x0]
; CHECK: ld1     {v1.2s}, [x1]
; CHECK: ld1     {v2.2s}, [x2]
; CHECK: st1     {v0.2s}, [x4]
; CHECK: st1     {v1.2s}, [x5]
; CHECK: st1     {v2.2s}, [x6]
; BE-STRICT-ALIGN-LABEL: align2vf32
; BE-STRICT-ALIGN: ldrb 
; BE-STRICT-ALIGN: ldrh    
; BE-STRICT-ALIGN: ldr
; BE-STRICT-ALIGN: strb    
; BE-STRICT-ALIGN: strh    
; BE-STRICT-ALIGN: str
entry:
  %val.byte  = load <2 x float>* %head.byte,  align 1
  %val.half  = load <2 x float>* %head.half,  align 2
  %val.word  = load <2 x float>* %head.word,  align 4
  store <2 x float> %val.byte,  <2 x float>* %tail.byte,  align 1
  store <2 x float> %val.half,  <2 x float>* %tail.half,  align 2
  store <2 x float> %val.word,  <2 x float>* %tail.word,  align 4
  ret <2 x float> %val.byte
}

; RUN: llc < %s -mcpu=x86-64 -mattr=+avx | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mcpu=x86-64 -mattr=+avx2 | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX2

target triple = "x86_64-unknown-unknown"

define <4 x i64> @testv4i64(<4 x i64> %in) {
; AVX1-LABEL: testv4i64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovaps {{.*#+}} xmm2 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX1-NEXT:    vandps %xmm2, %xmm1, %xmm3
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm4 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX1-NEXT:    vpshufb %xmm3, %xmm4, %xmm3
; AVX1-NEXT:    vpsrlw $4, %xmm1, %xmm1
; AVX1-NEXT:    vpand %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm4, %xmm1
; AVX1-NEXT:    vpaddb %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpxor %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpsadbw %xmm1, %xmm3, %xmm1
; AVX1-NEXT:    vandps %xmm2, %xmm0, %xmm5
; AVX1-NEXT:    vpshufb %xmm5, %xmm4, %xmm5
; AVX1-NEXT:    vpsrlw $4, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpshufb %xmm0, %xmm4, %xmm0
; AVX1-NEXT:    vpaddb %xmm5, %xmm0, %xmm0
; AVX1-NEXT:    vpsadbw %xmm0, %xmm3, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv4i64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm1 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm2
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm3 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX2-NEXT:    vpshufb %ymm2, %ymm3, %ymm2
; AVX2-NEXT:    vpsrlw $4, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpshufb %ymm0, %ymm3, %ymm0
; AVX2-NEXT:    vpaddb %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpxor %ymm1, %ymm1, %ymm1
; AVX2-NEXT:    vpsadbw %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %out = call <4 x i64> @llvm.ctpop.v4i64(<4 x i64> %in)
  ret <4 x i64> %out
}

define <8 x i32> @testv8i32(<8 x i32> %in) {
; AVX1-LABEL: testv8i32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovaps {{.*#+}} xmm2 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX1-NEXT:    vandps %xmm2, %xmm1, %xmm3
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm4 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX1-NEXT:    vpshufb %xmm3, %xmm4, %xmm3
; AVX1-NEXT:    vpsrlw $4, %xmm1, %xmm1
; AVX1-NEXT:    vpand %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm4, %xmm1
; AVX1-NEXT:    vpaddb %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpxor %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpunpckhdq {{.*#+}} xmm5 = xmm1[2],xmm3[2],xmm1[3],xmm3[3]
; AVX1-NEXT:    vpsadbw %xmm5, %xmm3, %xmm5
; AVX1-NEXT:    vpunpckldq {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1]
; AVX1-NEXT:    vpsadbw %xmm1, %xmm3, %xmm1
; AVX1-NEXT:    vpackuswb %xmm5, %xmm1, %xmm1
; AVX1-NEXT:    vandps %xmm2, %xmm0, %xmm5
; AVX1-NEXT:    vpshufb %xmm5, %xmm4, %xmm5
; AVX1-NEXT:    vpsrlw $4, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpshufb %xmm0, %xmm4, %xmm0
; AVX1-NEXT:    vpaddb %xmm5, %xmm0, %xmm0
; AVX1-NEXT:    vpunpckhdq {{.*#+}} xmm2 = xmm0[2],xmm3[2],xmm0[3],xmm3[3]
; AVX1-NEXT:    vpsadbw %xmm2, %xmm3, %xmm2
; AVX1-NEXT:    vpunpckldq {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1]
; AVX1-NEXT:    vpsadbw %xmm0, %xmm3, %xmm0
; AVX1-NEXT:    vpackuswb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv8i32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm1 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm2
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm3 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX2-NEXT:    vpshufb %ymm2, %ymm3, %ymm2
; AVX2-NEXT:    vpsrlw $4, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpshufb %ymm0, %ymm3, %ymm0
; AVX2-NEXT:    vpaddb %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpxor %ymm1, %ymm1, %ymm1
; AVX2-NEXT:    vpunpckhdq {{.*#+}} ymm2 = ymm0[2],ymm1[2],ymm0[3],ymm1[3],ymm0[6],ymm1[6],ymm0[7],ymm1[7]
; AVX2-NEXT:    vpsadbw %ymm2, %ymm1, %ymm2
; AVX2-NEXT:    vpunpckldq {{.*#+}} ymm0 = ymm0[0],ymm1[0],ymm0[1],ymm1[1],ymm0[4],ymm1[4],ymm0[5],ymm1[5]
; AVX2-NEXT:    vpsadbw %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    vpackuswb %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <8 x i32> @llvm.ctpop.v8i32(<8 x i32> %in)
  ret <8 x i32> %out
}

define <16 x i16> @testv16i16(<16 x i16> %in) {
; AVX1-LABEL: testv16i16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm1 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX1-NEXT:    vpand %xmm1, %xmm0, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX1-NEXT:    vpshufb %xmm2, %xmm3, %xmm2
; AVX1-NEXT:    vpsrlw $4, %xmm0, %xmm4
; AVX1-NEXT:    vpand %xmm1, %xmm4, %xmm4
; AVX1-NEXT:    vpshufb %xmm4, %xmm3, %xmm4
; AVX1-NEXT:    vpaddb %xmm2, %xmm4, %xmm2
; AVX1-NEXT:    vpsllw $8, %xmm2, %xmm4
; AVX1-NEXT:    vpaddb %xmm2, %xmm4, %xmm2
; AVX1-NEXT:    vpsrlw $8, %xmm2, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpand %xmm1, %xmm0, %xmm4
; AVX1-NEXT:    vpshufb %xmm4, %xmm3, %xmm4
; AVX1-NEXT:    vpsrlw $4, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vpshufb %xmm0, %xmm3, %xmm0
; AVX1-NEXT:    vpaddb %xmm4, %xmm0, %xmm0
; AVX1-NEXT:    vpsllw $8, %xmm0, %xmm1
; AVX1-NEXT:    vpaddb %xmm0, %xmm1, %xmm0
; AVX1-NEXT:    vpsrlw $8, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv16i16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm1 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm2
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm3 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX2-NEXT:    vpshufb %ymm2, %ymm3, %ymm2
; AVX2-NEXT:    vpsrlw $4, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpshufb %ymm0, %ymm3, %ymm0
; AVX2-NEXT:    vpaddb %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpsllw $8, %ymm0, %ymm1
; AVX2-NEXT:    vpaddb %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    vpsrlw $8, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <16 x i16> @llvm.ctpop.v16i16(<16 x i16> %in)
  ret <16 x i16> %out
}

define <32 x i8> @testv32i8(<32 x i8> %in) {
; AVX1-LABEL: testv32i8:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovaps {{.*#+}} xmm2 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX1-NEXT:    vandps %xmm2, %xmm1, %xmm3
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm4 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX1-NEXT:    vpshufb %xmm3, %xmm4, %xmm3
; AVX1-NEXT:    vpsrlw $4, %xmm1, %xmm1
; AVX1-NEXT:    vpand %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm4, %xmm1
; AVX1-NEXT:    vpaddb %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vandps %xmm2, %xmm0, %xmm3
; AVX1-NEXT:    vpshufb %xmm3, %xmm4, %xmm3
; AVX1-NEXT:    vpsrlw $4, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpshufb %xmm0, %xmm4, %xmm0
; AVX1-NEXT:    vpaddb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv32i8:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm1 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm2
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm3 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX2-NEXT:    vpshufb %ymm2, %ymm3, %ymm2
; AVX2-NEXT:    vpsrlw $4, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpshufb %ymm0, %ymm3, %ymm0
; AVX2-NEXT:    vpaddb %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <32 x i8> @llvm.ctpop.v32i8(<32 x i8> %in)
  ret <32 x i8> %out
}

declare <4 x i64> @llvm.ctpop.v4i64(<4 x i64>)
declare <8 x i32> @llvm.ctpop.v8i32(<8 x i32>)
declare <16 x i16> @llvm.ctpop.v16i16(<16 x i16>)
declare <32 x i8> @llvm.ctpop.v32i8(<32 x i8>)

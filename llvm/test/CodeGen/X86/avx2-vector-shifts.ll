; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

; AVX2 Logical Shift Left

define <16 x i16> @test_sllw_1(<16 x i16> %InVec) {
entry:
  %shl = shl <16 x i16> %InVec, <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_sllw_1:
; CHECK-NOT: vpsllw  $0, %ymm0, %ymm0
; CHECK: ret

define <16 x i16> @test_sllw_2(<16 x i16> %InVec) {
entry:
  %shl = shl <16 x i16> %InVec, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_sllw_2:
; CHECK: vpaddw  %ymm0, %ymm0, %ymm0
; CHECK: ret

define <16 x i16> @test_sllw_3(<16 x i16> %InVec) {
entry:
  %shl = shl <16 x i16> %InVec, <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_sllw_3:
; CHECK: vpsllw $15, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_slld_1(<8 x i32> %InVec) {
entry:
  %shl = shl <8 x i32> %InVec, <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_slld_1:
; CHECK-NOT: vpslld  $0, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_slld_2(<8 x i32> %InVec) {
entry:
  %shl = shl <8 x i32> %InVec, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_slld_2:
; CHECK: vpaddd  %ymm0, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_vpslld_var(i32 %shift) {
  %amt = insertelement <8 x i32> undef, i32 %shift, i32 0
  %tmp = shl <8 x i32> <i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199>, %amt
  ret <8 x i32> %tmp
}

; CHECK-LABEL: test_vpslld_var:
; CHECK: vpslld %xmm0, %ymm1, %ymm0
; CHECK: ret

define <8 x i32> @test_slld_3(<8 x i32> %InVec) {
entry:
  %shl = shl <8 x i32> %InVec, <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_slld_3:
; CHECK: vpslld $31, %ymm0, %ymm0
; CHECK: ret

define <4 x i64> @test_sllq_1(<4 x i64> %InVec) {
entry:
  %shl = shl <4 x i64> %InVec, <i64 0, i64 0, i64 0, i64 0>
  ret <4 x i64> %shl
}

; CHECK-LABEL: test_sllq_1:
; CHECK-NOT: vpsllq  $0, %ymm0, %ymm0
; CHECK: ret

define <4 x i64> @test_sllq_2(<4 x i64> %InVec) {
entry:
  %shl = shl <4 x i64> %InVec, <i64 1, i64 1, i64 1, i64 1>
  ret <4 x i64> %shl
}

; CHECK-LABEL: test_sllq_2:
; CHECK: vpaddq  %ymm0, %ymm0, %ymm0
; CHECK: ret

define <4 x i64> @test_sllq_3(<4 x i64> %InVec) {
entry:
  %shl = shl <4 x i64> %InVec, <i64 63, i64 63, i64 63, i64 63>
  ret <4 x i64> %shl
}

; CHECK-LABEL: test_sllq_3:
; CHECK: vpsllq $63, %ymm0, %ymm0
; CHECK: ret

; AVX2 Arithmetic Shift

define <16 x i16> @test_sraw_1(<16 x i16> %InVec) {
entry:
  %shl = ashr <16 x i16> %InVec, <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_sraw_1:
; CHECK-NOT: vpsraw  $0, %ymm0, %ymm0
; CHECK: ret

define <16 x i16> @test_sraw_2(<16 x i16> %InVec) {
entry:
  %shl = ashr <16 x i16> %InVec, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_sraw_2:
; CHECK: vpsraw  $1, %ymm0, %ymm0
; CHECK: ret

define <16 x i16> @test_sraw_3(<16 x i16> %InVec) {
entry:
  %shl = ashr <16 x i16> %InVec, <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_sraw_3:
; CHECK: vpsraw  $15, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_srad_1(<8 x i32> %InVec) {
entry:
  %shl = ashr <8 x i32> %InVec, <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_srad_1:
; CHECK-NOT: vpsrad  $0, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_srad_2(<8 x i32> %InVec) {
entry:
  %shl = ashr <8 x i32> %InVec, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_srad_2:
; CHECK: vpsrad  $1, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_srad_3(<8 x i32> %InVec) {
entry:
  %shl = ashr <8 x i32> %InVec, <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_srad_3:
; CHECK: vpsrad  $31, %ymm0, %ymm0
; CHECK: ret

; SSE Logical Shift Right

define <16 x i16> @test_srlw_1(<16 x i16> %InVec) {
entry:
  %shl = lshr <16 x i16> %InVec, <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_srlw_1:
; CHECK-NOT: vpsrlw  $0, %ymm0, %ymm0
; CHECK: ret

define <16 x i16> @test_srlw_2(<16 x i16> %InVec) {
entry:
  %shl = lshr <16 x i16> %InVec, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_srlw_2:
; CHECK: vpsrlw  $1, %ymm0, %ymm0
; CHECK: ret

define <16 x i16> @test_srlw_3(<16 x i16> %InVec) {
entry:
  %shl = lshr <16 x i16> %InVec, <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_srlw_3:
; CHECK: vpsrlw $15, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_srld_1(<8 x i32> %InVec) {
entry:
  %shl = lshr <8 x i32> %InVec, <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_srld_1:
; CHECK-NOT: vpsrld  $0, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_srld_2(<8 x i32> %InVec) {
entry:
  %shl = lshr <8 x i32> %InVec, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_srld_2:
; CHECK: vpsrld  $1, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_srld_3(<8 x i32> %InVec) {
entry:
  %shl = lshr <8 x i32> %InVec, <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_srld_3:
; CHECK: vpsrld $31, %ymm0, %ymm0
; CHECK: ret

define <4 x i64> @test_srlq_1(<4 x i64> %InVec) {
entry:
  %shl = lshr <4 x i64> %InVec, <i64 0, i64 0, i64 0, i64 0>
  ret <4 x i64> %shl
}

; CHECK-LABEL: test_srlq_1:
; CHECK-NOT: vpsrlq  $0, %ymm0, %ymm0
; CHECK: ret

define <4 x i64> @test_srlq_2(<4 x i64> %InVec) {
entry:
  %shl = lshr <4 x i64> %InVec, <i64 1, i64 1, i64 1, i64 1>
  ret <4 x i64> %shl
}

; CHECK-LABEL: test_srlq_2:
; CHECK: vpsrlq  $1, %ymm0, %ymm0
; CHECK: ret

define <4 x i64> @test_srlq_3(<4 x i64> %InVec) {
entry:
  %shl = lshr <4 x i64> %InVec, <i64 63, i64 63, i64 63, i64 63>
  ret <4 x i64> %shl
}

; CHECK-LABEL: test_srlq_3:
; CHECK: vpsrlq $63, %ymm0, %ymm0
; CHECK: ret

; CHECK-LABEL: @srl_trunc_and_v4i64
; CHECK: vpand
; CHECK-NEXT: vpsrlvd
; CHECK: ret
define <4 x i32> @srl_trunc_and_v4i64(<4 x i32> %x, <4 x i64> %y) nounwind {
  %and = and <4 x i64> %y, <i64 8, i64 8, i64 8, i64 8>
  %trunc = trunc <4 x i64> %and to <4 x i32>
  %sra = lshr <4 x i32> %x, %trunc
  ret <4 x i32> %sra
}

;
; Vectorized byte shifts
;

define <8 x i16> @shl_8i16(<8 x i16> %r, <8 x i16> %a) nounwind {
; CHECK-LABEL:  shl_8i16
; CHECK:        vpmovzxwd {{.*#+}} ymm1 = xmm1[0],zero,xmm1[1],zero,xmm1[2],zero,xmm1[3],zero,xmm1[4],zero,xmm1[5],zero,xmm1[6],zero,xmm1[7],zero
; CHECK-NEXT:   vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; CHECK-NEXT:   vpsllvd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:   vpshufb {{.*#+}} ymm0 = ymm0[0,1,4,5,8,9,12,13],zero,zero,zero,zero,zero,zero,zero,zero,ymm0[16,17,20,21,24,25,28,29],zero,zero,zero,zero,zero,zero,zero,zero
; CHECK-NEXT:   vpermq {{.*#+}} ymm0 = ymm0[0,2,2,3]
; CHECK:        retq
  %shl = shl <8 x i16> %r, %a
  ret <8 x i16> %shl
}

define <16 x i16> @shl_16i16(<16 x i16> %r, <16 x i16> %a) nounwind {
; CHECK-LABEL:  shl_16i16
; CHECK:        vpxor %ymm2, %ymm2, %ymm2
; CHECK-NEXT:   vpunpckhwd {{.*#+}} ymm3 = ymm1[4],ymm2[4],ymm1[5],ymm2[5],ymm1[6],ymm2[6],ymm1[7],ymm2[7],ymm1[12],ymm2[12],ymm1[13],ymm2[13],ymm1[14],ymm2[14],ymm1[15],ymm2[15]
; CHECK-NEXT:   vpunpckhwd {{.*#+}} ymm4 = ymm0[4,4,5,5,6,6,7,7,12,12,13,13,14,14,15,15]
; CHECK-NEXT:   vpsllvd %ymm3, %ymm4, %ymm3
; CHECK-NEXT:   vpsrld $16, %ymm3, %ymm3
; CHECK-NEXT:   vpunpcklwd {{.*#+}} ymm1 = ymm1[0],ymm2[0],ymm1[1],ymm2[1],ymm1[2],ymm2[2],ymm1[3],ymm2[3],ymm1[8],ymm2[8],ymm1[9],ymm2[9],ymm1[10],ymm2[10],ymm1[11],ymm2[11]
; CHECK-NEXT:   vpunpcklwd {{.*#+}} ymm0 = ymm0[0,0,1,1,2,2,3,3,8,8,9,9,10,10,11,11]
; CHECK-NEXT:   vpsllvd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:   vpsrld $16, %ymm0, %ymm0
; CHECK-NEXT:   vpackusdw %ymm3, %ymm0, %ymm0
; CHECK-NEXT:   retq
  %shl = shl <16 x i16> %r, %a
  ret <16 x i16> %shl
}

define <32 x i8> @shl_32i8(<32 x i8> %r, <32 x i8> %a) nounwind {
; CHECK-LABEL:  shl_32i8
; CHECK:        vextracti128 $1, %ymm0, %xmm3
; CHECK-NEXT:   vpsllw $4, %xmm3, %xmm2
; CHECK-NEXT:   vmovdqa  {{.*#+}} xmm8 = [240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240]
; CHECK-NEXT:   vpand %xmm8, %xmm2, %xmm5
; CHECK-NEXT:   vextracti128 $1, %ymm1, %xmm2
; CHECK-NEXT:   vpsllw $5, %xmm2, %xmm2
; CHECK-NEXT:   vmovdqa  {{.*#+}} xmm9 = [224,224,224,224,224,224,224,224,224,224,224,224,224,224,224,224]
; CHECK-NEXT:   vpand %xmm9, %xmm2, %xmm7
; CHECK-NEXT:   vmovdqa {{.*#+}} xmm2 = [128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128]
; CHECK-NEXT:   vpand %xmm7, %xmm2, %xmm4
; CHECK-NEXT:   vpcmpeqb %xmm2, %xmm4, %xmm4
; CHECK-NEXT:   vpblendvb %xmm4, %xmm5, %xmm3, %xmm3
; CHECK-NEXT:   vpsllw $2, %xmm3, %xmm4
; CHECK-NEXT:   vmovdqa  {{.*#+}} xmm5 = [252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252]
; CHECK-NEXT:   vpand %xmm5, %xmm4, %xmm4
; CHECK-NEXT:   vpaddb %xmm7, %xmm7, %xmm7
; CHECK-NEXT:   vpand %xmm7, %xmm2, %xmm6
; CHECK-NEXT:   vpcmpeqb %xmm2, %xmm6, %xmm6
; CHECK-NEXT:   vpblendvb %xmm6, %xmm4, %xmm3, %xmm3
; CHECK-NEXT:   vpaddb %xmm3, %xmm3, %xmm4
; CHECK-NEXT:   vpaddb %xmm7, %xmm7, %xmm6
; CHECK-NEXT:   vpand %xmm6, %xmm2, %xmm6
; CHECK-NEXT:   vpcmpeqb %xmm2, %xmm6, %xmm6
; CHECK-NEXT:   vpblendvb %xmm6, %xmm4, %xmm3, %xmm3
; CHECK-NEXT:   vpsllw $4, %xmm0, %xmm4
; CHECK-NEXT:   vpand %xmm8, %xmm4, %xmm4
; CHECK-NEXT:   vpsllw $5, %xmm1, %xmm1
; CHECK-NEXT:   vpand %xmm9, %xmm1, %xmm1
; CHECK-NEXT:   vpand %xmm1, %xmm2, %xmm6
; CHECK-NEXT:   vpcmpeqb %xmm2, %xmm6, %xmm6
; CHECK-NEXT:   vpblendvb %xmm6, %xmm4, %xmm0, %xmm0
; CHECK-NEXT:   vpsllw $2, %xmm0, %xmm4
; CHECK-NEXT:   vpand %xmm5, %xmm4, %xmm4
; CHECK-NEXT:   vpaddb %xmm1, %xmm1, %xmm1
; CHECK-NEXT:   vpand %xmm1, %xmm2, %xmm5
; CHECK-NEXT:   vpcmpeqb %xmm2, %xmm5, %xmm5
; CHECK-NEXT:   vpblendvb %xmm5, %xmm4, %xmm0, %xmm0
; CHECK-NEXT:   vpaddb %xmm0, %xmm0, %xmm4
; CHECK-NEXT:   vpaddb %xmm1, %xmm1, %xmm1
; CHECK-NEXT:   vpand %xmm1, %xmm2, %xmm1
; CHECK-NEXT:   vpcmpeqb %xmm2, %xmm1, %xmm1
; CHECK-NEXT:   vpblendvb %xmm1, %xmm4, %xmm0, %xmm0
; CHECK-NEXT:   vinserti128 $1, %xmm3, %ymm0, %ymm0
; CHECK-NEXT:   retq
  %shl = shl <32 x i8> %r, %a
  ret <32 x i8> %shl
}

define <8 x i16> @ashr_8i16(<8 x i16> %r, <8 x i16> %a) nounwind {
; CHECK-LABEL:  ashr_8i16
; CHECK:        vpmovzxwd {{.*#+}} ymm1 = xmm1[0],zero,xmm1[1],zero,xmm1[2],zero,xmm1[3],zero,xmm1[4],zero,xmm1[5],zero,xmm1[6],zero,xmm1[7],zero
; CHECK-NEXT:   vpmovsxwd %xmm0, %ymm0
; CHECK-NEXT:   vpsravd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:   vpshufb {{.*#+}} ymm0 = ymm0[0,1,4,5,8,9,12,13],zero,zero,zero,zero,zero,zero,zero,zero,ymm0[16,17,20,21,24,25,28,29],zero,zero,zero,zero,zero,zero,zero,zero
; CHECK-NEXT:   vpermq {{.*#+}} ymm0 = ymm0[0,2,2,3]
; CHECK:        retq
  %ashr = ashr <8 x i16> %r, %a
  ret <8 x i16> %ashr
}

define <16 x i16> @ashr_16i16(<16 x i16> %r, <16 x i16> %a) nounwind {
; CHECK-LABEL:  ashr_16i16
; CHECK:        vpxor %ymm2, %ymm2, %ymm2
; CHECK-NEXT:   vpunpckhwd {{.*#+}} ymm3 = ymm1[4],ymm2[4],ymm1[5],ymm2[5],ymm1[6],ymm2[6],ymm1[7],ymm2[7],ymm1[12],ymm2[12],ymm1[13],ymm2[13],ymm1[14],ymm2[14],ymm1[15],ymm2[15]
; CHECK-NEXT:   vpunpckhwd {{.*#+}} ymm4 = ymm0[4,4,5,5,6,6,7,7,12,12,13,13,14,14,15,15]
; CHECK-NEXT:   vpsravd %ymm3, %ymm4, %ymm3
; CHECK-NEXT:   vpsrld $16, %ymm3, %ymm3
; CHECK-NEXT:   vpunpcklwd {{.*#+}} ymm1 = ymm1[0],ymm2[0],ymm1[1],ymm2[1],ymm1[2],ymm2[2],ymm1[3],ymm2[3],ymm1[8],ymm2[8],ymm1[9],ymm2[9],ymm1[10],ymm2[10],ymm1[11],ymm2[11]
; CHECK-NEXT:   vpunpcklwd {{.*#+}} ymm0 = ymm0[0,0,1,1,2,2,3,3,8,8,9,9,10,10,11,11]
; CHECK-NEXT:   vpsravd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:   vpsrld $16, %ymm0, %ymm0
; CHECK-NEXT:   vpackusdw %ymm3, %ymm0, %ymm0
; CHECK-NEXT:   retq
  %ashr = ashr <16 x i16> %r, %a
  ret <16 x i16> %ashr
}

define <32 x i8> @ashr_32i8(<32 x i8> %r, <32 x i8> %a) nounwind {
; CHECK-LABEL:  ashr_32i8
; CHECK:        vextracti128 $1, %ymm1, %xmm2
; CHECK-NEXT:   vpextrb $1, %xmm2, %ecx
; CHECK-NEXT:   vextracti128 $1, %ymm0, %xmm3
; CHECK-NEXT:   vpextrb $1, %xmm3, %eax
; CHECK-NEXT:   sarb %cl, %al
; CHECK-NEXT:   vpextrb $0, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $0, %xmm3, %edx
; CHECK-NEXT:   sarb %cl, %dl
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   movzbl %dl, %edx
; CHECK-NEXT:   vpextrb $2, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $2, %xmm3, %esi
; CHECK-NEXT:   sarb %cl, %sil
; CHECK-NEXT:   vmovd %edx, %xmm4
; CHECK-NEXT:   vpinsrb $1, %eax, %xmm4, %xmm4
; CHECK-NEXT:   movzbl %sil, %eax
; CHECK-NEXT:   vpextrb $3, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $3, %xmm3, %edx
; CHECK-NEXT:   sarb %cl, %dl
; CHECK-NEXT:   vpinsrb $2, %eax, %xmm4, %xmm4
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpinsrb $3, %eax, %xmm4, %xmm4
; CHECK-NEXT:   vpextrb $4, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $4, %xmm3, %eax
; CHECK-NEXT:   sarb %cl, %al
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $4, %eax, %xmm4, %xmm4
; CHECK-NEXT:   vpextrb $5, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $5, %xmm3, %eax
; CHECK-NEXT:   sarb %cl, %al
; CHECK-NEXT:   vpextrb $6, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $6, %xmm3, %edx
; CHECK-NEXT:   sarb %cl, %dl
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $5, %eax, %xmm4, %xmm4
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpextrb $7, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $7, %xmm3, %edx
; CHECK-NEXT:   sarb %cl, %dl
; CHECK-NEXT:   vpinsrb $6, %eax, %xmm4, %xmm4
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpinsrb $7, %eax, %xmm4, %xmm4
; CHECK-NEXT:   vpextrb $8, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $8, %xmm3, %eax
; CHECK-NEXT:   sarb %cl, %al
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $8, %eax, %xmm4, %xmm4
; CHECK-NEXT:   vpextrb $9, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $9, %xmm3, %eax
; CHECK-NEXT:   sarb %cl, %al
; CHECK-NEXT:   vpextrb $10, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $10, %xmm3, %edx
; CHECK-NEXT:   sarb %cl, %dl
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $9, %eax, %xmm4, %xmm4
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpextrb $11, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $11, %xmm3, %edx
; CHECK-NEXT:   sarb %cl, %dl
; CHECK-NEXT:   vpinsrb $10, %eax, %xmm4, %xmm4
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpinsrb $11, %eax, %xmm4, %xmm4
; CHECK-NEXT:   vpextrb $12, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $12, %xmm3, %eax
; CHECK-NEXT:   sarb %cl, %al
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $12, %eax, %xmm4, %xmm4
; CHECK-NEXT:   vpextrb $13, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $13, %xmm3, %eax
; CHECK-NEXT:   sarb %cl, %al
; CHECK-NEXT:   vpextrb $14, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $14, %xmm3, %edx
; CHECK-NEXT:   sarb %cl, %dl
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $13, %eax, %xmm4, %xmm4
; CHECK-NEXT:   vpextrb $15, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $15, %xmm3, %eax
; CHECK-NEXT:   sarb %cl, %al
; CHECK-NEXT:   vpextrb $1, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $1, %xmm0, %esi
; CHECK-NEXT:   sarb %cl, %sil
; CHECK-NEXT:   movzbl %dl, %ecx
; CHECK-NEXT:   vpinsrb $14, %ecx, %xmm4, %xmm2
; CHECK-NEXT:   vpextrb $0, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $0, %xmm0, %edx
; CHECK-NEXT:   sarb %cl, %dl
; CHECK-NEXT:   vpextrb $2, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $2, %xmm0, %edi
; CHECK-NEXT:   sarb %cl, %dil
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $15, %eax, %xmm2, %xmm2
; CHECK-NEXT:   movzbl %sil, %eax
; CHECK-NEXT:   movzbl %dl, %ecx
; CHECK-NEXT:   vmovd %ecx, %xmm3
; CHECK-NEXT:   vpinsrb $1, %eax, %xmm3, %xmm3
; CHECK-NEXT:   movzbl %dil, %eax
; CHECK-NEXT:   vpextrb $3, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $3, %xmm0, %edx
; CHECK-NEXT:   sarb %cl, %dl
; CHECK-NEXT:   vpinsrb $2, %eax, %xmm3, %xmm3
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpinsrb $3, %eax, %xmm3, %xmm3
; CHECK-NEXT:   vpextrb $4, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $4, %xmm0, %eax
; CHECK-NEXT:   sarb %cl, %al
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $4, %eax, %xmm3, %xmm3
; CHECK-NEXT:   vpextrb $5, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $5, %xmm0, %eax
; CHECK-NEXT:   sarb %cl, %al
; CHECK-NEXT:   vpextrb $6, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $6, %xmm0, %edx
; CHECK-NEXT:   sarb %cl, %dl
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $5, %eax, %xmm3, %xmm3
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpextrb $7, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $7, %xmm0, %edx
; CHECK-NEXT:   sarb %cl, %dl
; CHECK-NEXT:   vpinsrb $6, %eax, %xmm3, %xmm3
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpinsrb $7, %eax, %xmm3, %xmm3
; CHECK-NEXT:   vpextrb $8, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $8, %xmm0, %eax
; CHECK-NEXT:   sarb %cl, %al
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $8, %eax, %xmm3, %xmm3
; CHECK-NEXT:   vpextrb $9, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $9, %xmm0, %eax
; CHECK-NEXT:   sarb %cl, %al
; CHECK-NEXT:   vpextrb $10, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $10, %xmm0, %edx
; CHECK-NEXT:   sarb %cl, %dl
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $9, %eax, %xmm3, %xmm3
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpextrb $11, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $11, %xmm0, %edx
; CHECK-NEXT:   sarb %cl, %dl
; CHECK-NEXT:   vpinsrb $10, %eax, %xmm3, %xmm3
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpinsrb $11, %eax, %xmm3, %xmm3
; CHECK-NEXT:   vpextrb $12, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $12, %xmm0, %eax
; CHECK-NEXT:   sarb %cl, %al
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $12, %eax, %xmm3, %xmm3
; CHECK-NEXT:   vpextrb $13, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $13, %xmm0, %eax
; CHECK-NEXT:   sarb %cl, %al
; CHECK-NEXT:   vpextrb $14, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $14, %xmm0, %edx
; CHECK-NEXT:   sarb %cl, %dl
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $13, %eax, %xmm3, %xmm3
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpextrb $15, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $15, %xmm0, %edx
; CHECK-NEXT:   sarb %cl, %dl
; CHECK-NEXT:   vpinsrb $14, %eax, %xmm3, %xmm0
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpinsrb $15, %eax, %xmm0, %xmm0
; CHECK-NEXT:   vinserti128 $1, %xmm2, %ymm0, %ymm0
; CHECK-NEXT:   retq
  %ashr = ashr <32 x i8> %r, %a
  ret <32 x i8> %ashr
}

define <8 x i16> @lshr_8i16(<8 x i16> %r, <8 x i16> %a) nounwind {
; CHECK-LABEL:  lshr_8i16
; CHECK:        vpmovzxwd {{.*#+}} ymm1 = xmm1[0],zero,xmm1[1],zero,xmm1[2],zero,xmm1[3],zero,xmm1[4],zero,xmm1[5],zero,xmm1[6],zero,xmm1[7],zero
; CHECK-NEXT:   vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; CHECK-NEXT:   vpsrlvd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:   vpshufb {{.*#+}} ymm0 = ymm0[0,1,4,5,8,9,12,13],zero,zero,zero,zero,zero,zero,zero,zero,ymm0[16,17,20,21,24,25,28,29],zero,zero,zero,zero,zero,zero,zero,zero
; CHECK-NEXT:   vpermq {{.*#+}} ymm0 = ymm0[0,2,2,3]
; CHECK:        retq
  %lshr = lshr <8 x i16> %r, %a
  ret <8 x i16> %lshr
}

define <16 x i16> @lshr_16i16(<16 x i16> %r, <16 x i16> %a) nounwind {
; CHECK-LABEL:  lshr_16i16
; CHECK:        vpxor %ymm2, %ymm2, %ymm2
; CHECK-NEXT:   vpunpckhwd {{.*#+}} ymm3 = ymm1[4],ymm2[4],ymm1[5],ymm2[5],ymm1[6],ymm2[6],ymm1[7],ymm2[7],ymm1[12],ymm2[12],ymm1[13],ymm2[13],ymm1[14],ymm2[14],ymm1[15],ymm2[15]
; CHECK-NEXT:   vpunpckhwd {{.*#+}} ymm4 = ymm0[4,4,5,5,6,6,7,7,12,12,13,13,14,14,15,15]
; CHECK-NEXT:   vpsrlvd %ymm3, %ymm4, %ymm3
; CHECK-NEXT:   vpsrld $16, %ymm3, %ymm3
; CHECK-NEXT:   vpunpcklwd {{.*#+}} ymm1 = ymm1[0],ymm2[0],ymm1[1],ymm2[1],ymm1[2],ymm2[2],ymm1[3],ymm2[3],ymm1[8],ymm2[8],ymm1[9],ymm2[9],ymm1[10],ymm2[10],ymm1[11],ymm2[11]
; CHECK-NEXT:   vpunpcklwd {{.*#+}} ymm0 = ymm0[0,0,1,1,2,2,3,3,8,8,9,9,10,10,11,11]
; CHECK-NEXT:   vpsrlvd %ymm1, %ymm0, %ymm0
; CHECK-NEXT:   vpsrld $16, %ymm0, %ymm0
; CHECK-NEXT:   vpackusdw %ymm3, %ymm0, %ymm0
; CHECK-NEXT:   retq
  %lshr = lshr <16 x i16> %r, %a
  ret <16 x i16> %lshr
}

define <32 x i8> @lshr_32i8(<32 x i8> %r, <32 x i8> %a) nounwind {
; CHECK-LABEL:  lshr_32i8
; CHECK:        vextracti128 $1, %ymm1, %xmm2
; CHECK-NEXT:   vpextrb $1, %xmm2, %ecx
; CHECK-NEXT:   vextracti128 $1, %ymm0, %xmm3
; CHECK-NEXT:   vpextrb $1, %xmm3, %eax
; CHECK-NEXT:   shrb %cl, %al
; CHECK-NEXT:   vpextrb $0, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $0, %xmm3, %edx
; CHECK-NEXT:   shrb %cl, %dl
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   movzbl %dl, %edx
; CHECK-NEXT:   vpextrb $2, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $2, %xmm3, %esi
; CHECK-NEXT:   shrb %cl, %sil
; CHECK-NEXT:   vmovd %edx, %xmm4
; CHECK-NEXT:   vpinsrb $1, %eax, %xmm4, %xmm4
; CHECK-NEXT:   movzbl %sil, %eax
; CHECK-NEXT:   vpextrb $3, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $3, %xmm3, %edx
; CHECK-NEXT:   shrb %cl, %dl
; CHECK-NEXT:   vpinsrb $2, %eax, %xmm4, %xmm4
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpinsrb $3, %eax, %xmm4, %xmm4
; CHECK-NEXT:   vpextrb $4, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $4, %xmm3, %eax
; CHECK-NEXT:   shrb %cl, %al
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $4, %eax, %xmm4, %xmm4
; CHECK-NEXT:   vpextrb $5, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $5, %xmm3, %eax
; CHECK-NEXT:   shrb %cl, %al
; CHECK-NEXT:   vpextrb $6, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $6, %xmm3, %edx
; CHECK-NEXT:   shrb %cl, %dl
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $5, %eax, %xmm4, %xmm4
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpextrb $7, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $7, %xmm3, %edx
; CHECK-NEXT:   shrb %cl, %dl
; CHECK-NEXT:   vpinsrb $6, %eax, %xmm4, %xmm4
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpinsrb $7, %eax, %xmm4, %xmm4
; CHECK-NEXT:   vpextrb $8, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $8, %xmm3, %eax
; CHECK-NEXT:   shrb %cl, %al
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $8, %eax, %xmm4, %xmm4
; CHECK-NEXT:   vpextrb $9, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $9, %xmm3, %eax
; CHECK-NEXT:   shrb %cl, %al
; CHECK-NEXT:   vpextrb $10, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $10, %xmm3, %edx
; CHECK-NEXT:   shrb %cl, %dl
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $9, %eax, %xmm4, %xmm4
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpextrb $11, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $11, %xmm3, %edx
; CHECK-NEXT:   shrb %cl, %dl
; CHECK-NEXT:   vpinsrb $10, %eax, %xmm4, %xmm4
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpinsrb $11, %eax, %xmm4, %xmm4
; CHECK-NEXT:   vpextrb $12, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $12, %xmm3, %eax
; CHECK-NEXT:   shrb %cl, %al
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $12, %eax, %xmm4, %xmm4
; CHECK-NEXT:   vpextrb $13, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $13, %xmm3, %eax
; CHECK-NEXT:   shrb %cl, %al
; CHECK-NEXT:   vpextrb $14, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $14, %xmm3, %edx
; CHECK-NEXT:   shrb %cl, %dl
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $13, %eax, %xmm4, %xmm4
; CHECK-NEXT:   vpextrb $15, %xmm2, %ecx
; CHECK-NEXT:   vpextrb $15, %xmm3, %eax
; CHECK-NEXT:   shrb %cl, %al
; CHECK-NEXT:   vpextrb $1, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $1, %xmm0, %esi
; CHECK-NEXT:   shrb %cl, %sil
; CHECK-NEXT:   movzbl %dl, %ecx
; CHECK-NEXT:   vpinsrb $14, %ecx, %xmm4, %xmm2
; CHECK-NEXT:   vpextrb $0, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $0, %xmm0, %edx
; CHECK-NEXT:   shrb %cl, %dl
; CHECK-NEXT:   vpextrb $2, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $2, %xmm0, %edi
; CHECK-NEXT:   shrb %cl, %dil
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $15, %eax, %xmm2, %xmm2
; CHECK-NEXT:   movzbl %sil, %eax
; CHECK-NEXT:   movzbl %dl, %ecx
; CHECK-NEXT:   vmovd %ecx, %xmm3
; CHECK-NEXT:   vpinsrb $1, %eax, %xmm3, %xmm3
; CHECK-NEXT:   movzbl %dil, %eax
; CHECK-NEXT:   vpextrb $3, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $3, %xmm0, %edx
; CHECK-NEXT:   shrb %cl, %dl
; CHECK-NEXT:   vpinsrb $2, %eax, %xmm3, %xmm3
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpinsrb $3, %eax, %xmm3, %xmm3
; CHECK-NEXT:   vpextrb $4, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $4, %xmm0, %eax
; CHECK-NEXT:   shrb %cl, %al
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $4, %eax, %xmm3, %xmm3
; CHECK-NEXT:   vpextrb $5, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $5, %xmm0, %eax
; CHECK-NEXT:   shrb %cl, %al
; CHECK-NEXT:   vpextrb $6, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $6, %xmm0, %edx
; CHECK-NEXT:   shrb %cl, %dl
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $5, %eax, %xmm3, %xmm3
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpextrb $7, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $7, %xmm0, %edx
; CHECK-NEXT:   shrb %cl, %dl
; CHECK-NEXT:   vpinsrb $6, %eax, %xmm3, %xmm3
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpinsrb $7, %eax, %xmm3, %xmm3
; CHECK-NEXT:   vpextrb $8, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $8, %xmm0, %eax
; CHECK-NEXT:   shrb %cl, %al
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $8, %eax, %xmm3, %xmm3
; CHECK-NEXT:   vpextrb $9, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $9, %xmm0, %eax
; CHECK-NEXT:   shrb %cl, %al
; CHECK-NEXT:   vpextrb $10, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $10, %xmm0, %edx
; CHECK-NEXT:   shrb %cl, %dl
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $9, %eax, %xmm3, %xmm3
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpextrb $11, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $11, %xmm0, %edx
; CHECK-NEXT:   shrb %cl, %dl
; CHECK-NEXT:   vpinsrb $10, %eax, %xmm3, %xmm3
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpinsrb $11, %eax, %xmm3, %xmm3
; CHECK-NEXT:   vpextrb $12, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $12, %xmm0, %eax
; CHECK-NEXT:   shrb %cl, %al
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $12, %eax, %xmm3, %xmm3
; CHECK-NEXT:   vpextrb $13, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $13, %xmm0, %eax
; CHECK-NEXT:   shrb %cl, %al
; CHECK-NEXT:   vpextrb $14, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $14, %xmm0, %edx
; CHECK-NEXT:   shrb %cl, %dl
; CHECK-NEXT:   movzbl %al, %eax
; CHECK-NEXT:   vpinsrb $13, %eax, %xmm3, %xmm3
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpextrb $15, %xmm1, %ecx
; CHECK-NEXT:   vpextrb $15, %xmm0, %edx
; CHECK-NEXT:   shrb %cl, %dl
; CHECK-NEXT:   vpinsrb $14, %eax, %xmm3, %xmm0
; CHECK-NEXT:   movzbl %dl, %eax
; CHECK-NEXT:   vpinsrb $15, %eax, %xmm0, %xmm0
; CHECK-NEXT:   vinserti128 $1, %xmm2, %ymm0, %ymm0
; CHECK-NEXT:   retq
  %lshr = lshr <32 x i8> %r, %a
  ret <32 x i8> %lshr
}

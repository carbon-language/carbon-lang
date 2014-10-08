; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512bw --show-mc-encoding| FileCheck %s

define i64 @test_pcmpeq_b(<64 x i8> %a, <64 x i8> %b) {
; CHECK-LABEL: test_pcmpeq_b
; CHECK: vpcmpeqb %zmm1, %zmm0, %k0 ##
  %res = call i64 @llvm.x86.avx512.mask.pcmpeq.b.512(<64 x i8> %a, <64 x i8> %b, i64 -1)
  ret i64 %res
}

define i64 @test_mask_pcmpeq_b(<64 x i8> %a, <64 x i8> %b, i64 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_b
; CHECK: vpcmpeqb %zmm1, %zmm0, %k0 {%k1} ##
  %res = call i64 @llvm.x86.avx512.mask.pcmpeq.b.512(<64 x i8> %a, <64 x i8> %b, i64 %mask)
  ret i64 %res
}

declare i64 @llvm.x86.avx512.mask.pcmpeq.b.512(<64 x i8>, <64 x i8>, i64)

define i32 @test_pcmpeq_w(<32 x i16> %a, <32 x i16> %b) {
; CHECK-LABEL: test_pcmpeq_w
; CHECK: vpcmpeqw %zmm1, %zmm0, %k0 ##
  %res = call i32 @llvm.x86.avx512.mask.pcmpeq.w.512(<32 x i16> %a, <32 x i16> %b, i32 -1)
  ret i32 %res
}

define i32 @test_mask_pcmpeq_w(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_w
; CHECK: vpcmpeqw %zmm1, %zmm0, %k0 {%k1} ##
  %res = call i32 @llvm.x86.avx512.mask.pcmpeq.w.512(<32 x i16> %a, <32 x i16> %b, i32 %mask)
  ret i32 %res
}

declare i32 @llvm.x86.avx512.mask.pcmpeq.w.512(<32 x i16>, <32 x i16>, i32)

define i64 @test_pcmpgt_b(<64 x i8> %a, <64 x i8> %b) {
; CHECK-LABEL: test_pcmpgt_b
; CHECK: vpcmpgtb %zmm1, %zmm0, %k0 ##
  %res = call i64 @llvm.x86.avx512.mask.pcmpgt.b.512(<64 x i8> %a, <64 x i8> %b, i64 -1)
  ret i64 %res
}

define i64 @test_mask_pcmpgt_b(<64 x i8> %a, <64 x i8> %b, i64 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_b
; CHECK: vpcmpgtb %zmm1, %zmm0, %k0 {%k1} ##
  %res = call i64 @llvm.x86.avx512.mask.pcmpgt.b.512(<64 x i8> %a, <64 x i8> %b, i64 %mask)
  ret i64 %res
}

declare i64 @llvm.x86.avx512.mask.pcmpgt.b.512(<64 x i8>, <64 x i8>, i64)

define i32 @test_pcmpgt_w(<32 x i16> %a, <32 x i16> %b) {
; CHECK-LABEL: test_pcmpgt_w
; CHECK: vpcmpgtw %zmm1, %zmm0, %k0 ##
  %res = call i32 @llvm.x86.avx512.mask.pcmpgt.w.512(<32 x i16> %a, <32 x i16> %b, i32 -1)
  ret i32 %res
}

define i32 @test_mask_pcmpgt_w(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_w
; CHECK: vpcmpgtw %zmm1, %zmm0, %k0 {%k1} ##
  %res = call i32 @llvm.x86.avx512.mask.pcmpgt.w.512(<32 x i16> %a, <32 x i16> %b, i32 %mask)
  ret i32 %res
}

declare i32 @llvm.x86.avx512.mask.pcmpgt.w.512(<32 x i16>, <32 x i16>, i32)

define <8 x i64> @test_cmp_b_512(<64 x i8> %a0, <64 x i8> %a1) {
; CHECK_LABEL: test_cmp_b_512
; CHECK: vpcmpeqb %zmm1, %zmm0, %k0 ##
  %res0 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 0, i64 -1)
  %vec0 = insertelement <8 x i64> undef, i64 %res0, i32 0
; CHECK: vpcmpltb %zmm1, %zmm0, %k0 ##
  %res1 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 1, i64 -1)
  %vec1 = insertelement <8 x i64> %vec0, i64 %res1, i32 1
; CHECK: vpcmpleb %zmm1, %zmm0, %k0 ##
  %res2 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 2, i64 -1)
  %vec2 = insertelement <8 x i64> %vec1, i64 %res2, i32 2
; CHECK: vpcmpunordb %zmm1, %zmm0, %k0 ##
  %res3 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 3, i64 -1)
  %vec3 = insertelement <8 x i64> %vec2, i64 %res3, i32 3
; CHECK: vpcmpneqb %zmm1, %zmm0, %k0 ##
  %res4 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 4, i64 -1)
  %vec4 = insertelement <8 x i64> %vec3, i64 %res4, i32 4
; CHECK: vpcmpnltb %zmm1, %zmm0, %k0 ##
  %res5 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 5, i64 -1)
  %vec5 = insertelement <8 x i64> %vec4, i64 %res5, i32 5
; CHECK: vpcmpnleb %zmm1, %zmm0, %k0 ##
  %res6 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 6, i64 -1)
  %vec6 = insertelement <8 x i64> %vec5, i64 %res6, i32 6
; CHECK: vpcmpordb %zmm1, %zmm0, %k0 ##
  %res7 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 7, i64 -1)
  %vec7 = insertelement <8 x i64> %vec6, i64 %res7, i32 7
  ret <8 x i64> %vec7
}

define <8 x i64> @test_mask_cmp_b_512(<64 x i8> %a0, <64 x i8> %a1, i64 %mask) {
; CHECK_LABEL: test_mask_cmp_b_512
; CHECK: vpcmpeqb %zmm1, %zmm0, %k0 {%k1} ##
  %res0 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 0, i64 %mask)
  %vec0 = insertelement <8 x i64> undef, i64 %res0, i32 0
; CHECK: vpcmpltb %zmm1, %zmm0, %k0 {%k1} ##
  %res1 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 1, i64 %mask)
  %vec1 = insertelement <8 x i64> %vec0, i64 %res1, i32 1
; CHECK: vpcmpleb %zmm1, %zmm0, %k0 {%k1} ##
  %res2 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 2, i64 %mask)
  %vec2 = insertelement <8 x i64> %vec1, i64 %res2, i32 2
; CHECK: vpcmpunordb %zmm1, %zmm0, %k0 {%k1} ##
  %res3 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 3, i64 %mask)
  %vec3 = insertelement <8 x i64> %vec2, i64 %res3, i32 3
; CHECK: vpcmpneqb %zmm1, %zmm0, %k0 {%k1} ##
  %res4 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 4, i64 %mask)
  %vec4 = insertelement <8 x i64> %vec3, i64 %res4, i32 4
; CHECK: vpcmpnltb %zmm1, %zmm0, %k0 {%k1} ##
  %res5 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 5, i64 %mask)
  %vec5 = insertelement <8 x i64> %vec4, i64 %res5, i32 5
; CHECK: vpcmpnleb %zmm1, %zmm0, %k0 {%k1} ##
  %res6 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 6, i64 %mask)
  %vec6 = insertelement <8 x i64> %vec5, i64 %res6, i32 6
; CHECK: vpcmpordb %zmm1, %zmm0, %k0 {%k1} ##
  %res7 = call i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 7, i64 %mask)
  %vec7 = insertelement <8 x i64> %vec6, i64 %res7, i32 7
  ret <8 x i64> %vec7
}

declare i64 @llvm.x86.avx512.mask.cmp.b.512(<64 x i8>, <64 x i8>, i32, i64) nounwind readnone

define <8 x i64> @test_ucmp_b_512(<64 x i8> %a0, <64 x i8> %a1) {
; CHECK_LABEL: test_ucmp_b_512
; CHECK: vpcmpequb %zmm1, %zmm0, %k0 ##
  %res0 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 0, i64 -1)
  %vec0 = insertelement <8 x i64> undef, i64 %res0, i32 0
; CHECK: vpcmpltub %zmm1, %zmm0, %k0 ##
  %res1 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 1, i64 -1)
  %vec1 = insertelement <8 x i64> %vec0, i64 %res1, i32 1
; CHECK: vpcmpleub %zmm1, %zmm0, %k0 ##
  %res2 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 2, i64 -1)
  %vec2 = insertelement <8 x i64> %vec1, i64 %res2, i32 2
; CHECK: vpcmpunordub %zmm1, %zmm0, %k0 ##
  %res3 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 3, i64 -1)
  %vec3 = insertelement <8 x i64> %vec2, i64 %res3, i32 3
; CHECK: vpcmpnequb %zmm1, %zmm0, %k0 ##
  %res4 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 4, i64 -1)
  %vec4 = insertelement <8 x i64> %vec3, i64 %res4, i32 4
; CHECK: vpcmpnltub %zmm1, %zmm0, %k0 ##
  %res5 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 5, i64 -1)
  %vec5 = insertelement <8 x i64> %vec4, i64 %res5, i32 5
; CHECK: vpcmpnleub %zmm1, %zmm0, %k0 ##
  %res6 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 6, i64 -1)
  %vec6 = insertelement <8 x i64> %vec5, i64 %res6, i32 6
; CHECK: vpcmpordub %zmm1, %zmm0, %k0 ##
  %res7 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 7, i64 -1)
  %vec7 = insertelement <8 x i64> %vec6, i64 %res7, i32 7
  ret <8 x i64> %vec7
}

define <8 x i64> @test_mask_x86_avx512_ucmp_b_512(<64 x i8> %a0, <64 x i8> %a1, i64 %mask) {
; CHECK_LABEL: test_mask_ucmp_b_512
; CHECK: vpcmpequb %zmm1, %zmm0, %k0 {%k1} ##
  %res0 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 0, i64 %mask)
  %vec0 = insertelement <8 x i64> undef, i64 %res0, i32 0
; CHECK: vpcmpltub %zmm1, %zmm0, %k0 {%k1} ##
  %res1 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 1, i64 %mask)
  %vec1 = insertelement <8 x i64> %vec0, i64 %res1, i32 1
; CHECK: vpcmpleub %zmm1, %zmm0, %k0 {%k1} ##
  %res2 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 2, i64 %mask)
  %vec2 = insertelement <8 x i64> %vec1, i64 %res2, i32 2
; CHECK: vpcmpunordub %zmm1, %zmm0, %k0 {%k1} ##
  %res3 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 3, i64 %mask)
  %vec3 = insertelement <8 x i64> %vec2, i64 %res3, i32 3
; CHECK: vpcmpnequb %zmm1, %zmm0, %k0 {%k1} ##
  %res4 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 4, i64 %mask)
  %vec4 = insertelement <8 x i64> %vec3, i64 %res4, i32 4
; CHECK: vpcmpnltub %zmm1, %zmm0, %k0 {%k1} ##
  %res5 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 5, i64 %mask)
  %vec5 = insertelement <8 x i64> %vec4, i64 %res5, i32 5
; CHECK: vpcmpnleub %zmm1, %zmm0, %k0 {%k1} ##
  %res6 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 6, i64 %mask)
  %vec6 = insertelement <8 x i64> %vec5, i64 %res6, i32 6
; CHECK: vpcmpordub %zmm1, %zmm0, %k0 {%k1} ##
  %res7 = call i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> %a0, <64 x i8> %a1, i32 7, i64 %mask)
  %vec7 = insertelement <8 x i64> %vec6, i64 %res7, i32 7
  ret <8 x i64> %vec7
}

declare i64 @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8>, <64 x i8>, i32, i64) nounwind readnone

define <8 x i32> @test_cmp_w_512(<32 x i16> %a0, <32 x i16> %a1) {
; CHECK_LABEL: test_cmp_w_512
; CHECK: vpcmpeqw %zmm1, %zmm0, %k0 ##
  %res0 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 0, i32 -1)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltw %zmm1, %zmm0, %k0 ##
  %res1 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 1, i32 -1)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmplew %zmm1, %zmm0, %k0 ##
  %res2 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 2, i32 -1)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunordw %zmm1, %zmm0, %k0 ##
  %res3 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 3, i32 -1)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpneqw %zmm1, %zmm0, %k0 ##
  %res4 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 4, i32 -1)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltw %zmm1, %zmm0, %k0 ##
  %res5 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 5, i32 -1)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnlew %zmm1, %zmm0, %k0 ##
  %res6 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 6, i32 -1)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmpordw %zmm1, %zmm0, %k0 ##
  %res7 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 7, i32 -1)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

define <8 x i32> @test_mask_cmp_w_512(<32 x i16> %a0, <32 x i16> %a1, i32 %mask) {
; CHECK_LABEL: test_mask_cmp_w_512
; CHECK: vpcmpeqw %zmm1, %zmm0, %k0 {%k1} ##
  %res0 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 0, i32 %mask)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltw %zmm1, %zmm0, %k0 {%k1} ##
  %res1 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 1, i32 %mask)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmplew %zmm1, %zmm0, %k0 {%k1} ##
  %res2 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 2, i32 %mask)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunordw %zmm1, %zmm0, %k0 {%k1} ##
  %res3 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 3, i32 %mask)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpneqw %zmm1, %zmm0, %k0 {%k1} ##
  %res4 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 4, i32 %mask)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltw %zmm1, %zmm0, %k0 {%k1} ##
  %res5 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 5, i32 %mask)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnlew %zmm1, %zmm0, %k0 {%k1} ##
  %res6 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 6, i32 %mask)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmpordw %zmm1, %zmm0, %k0 {%k1} ##
  %res7 = call i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 7, i32 %mask)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

declare i32 @llvm.x86.avx512.mask.cmp.w.512(<32 x i16>, <32 x i16>, i32, i32) nounwind readnone

define <8 x i32> @test_ucmp_w_512(<32 x i16> %a0, <32 x i16> %a1) {
; CHECK_LABEL: test_ucmp_w_512
; CHECK: vpcmpequw %zmm1, %zmm0, %k0 ##
  %res0 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 0, i32 -1)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltuw %zmm1, %zmm0, %k0 ##
  %res1 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 1, i32 -1)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmpleuw %zmm1, %zmm0, %k0 ##
  %res2 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 2, i32 -1)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunorduw %zmm1, %zmm0, %k0 ##
  %res3 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 3, i32 -1)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpnequw %zmm1, %zmm0, %k0 ##
  %res4 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 4, i32 -1)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltuw %zmm1, %zmm0, %k0 ##
  %res5 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 5, i32 -1)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnleuw %zmm1, %zmm0, %k0 ##
  %res6 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 6, i32 -1)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmporduw %zmm1, %zmm0, %k0 ##
  %res7 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 7, i32 -1)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

define <8 x i32> @test_mask_ucmp_w_512(<32 x i16> %a0, <32 x i16> %a1, i32 %mask) {
; CHECK_LABEL: test_mask_ucmp_w_512
; CHECK: vpcmpequw %zmm1, %zmm0, %k0 {%k1} ##
  %res0 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 0, i32 %mask)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltuw %zmm1, %zmm0, %k0 {%k1} ##
  %res1 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 1, i32 %mask)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmpleuw %zmm1, %zmm0, %k0 {%k1} ##
  %res2 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 2, i32 %mask)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunorduw %zmm1, %zmm0, %k0 {%k1} ##
  %res3 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 3, i32 %mask)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpnequw %zmm1, %zmm0, %k0 {%k1} ##
  %res4 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 4, i32 %mask)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltuw %zmm1, %zmm0, %k0 {%k1} ##
  %res5 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 5, i32 %mask)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnleuw %zmm1, %zmm0, %k0 {%k1} ##
  %res6 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 6, i32 %mask)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmporduw %zmm1, %zmm0, %k0 {%k1} ##
  %res7 = call i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> %a0, <32 x i16> %a1, i32 7, i32 %mask)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

declare i32 @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16>, <32 x i16>, i32, i32) nounwind readnone

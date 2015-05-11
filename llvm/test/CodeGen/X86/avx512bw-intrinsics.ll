; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=skx --show-mc-encoding| FileCheck %s

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

; CHECK-LABEL: test_x86_mask_blend_b_256
; CHECK: vpblendmb
define <32 x i8> @test_x86_mask_blend_b_256(i32 %a0, <32 x i8> %a1, <32 x i8> %a2) {
  %res = call <32 x i8> @llvm.x86.avx512.mask.blend.b.256(<32 x i8> %a1, <32 x i8> %a2, i32 %a0) ; <<32 x i8>> [#uses=1]
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.x86.avx512.mask.blend.b.256(<32 x i8>, <32 x i8>, i32) nounwind readonly

; CHECK-LABEL: test_x86_mask_blend_w_256
define <16 x i16> @test_x86_mask_blend_w_256(i16 %mask, <16 x i16> %a1, <16 x i16> %a2) {
  ; CHECK: vpblendmw
  %res = call <16 x i16> @llvm.x86.avx512.mask.blend.w.256(<16 x i16> %a1, <16 x i16> %a2, i16 %mask) ; <<16 x i16>> [#uses=1]
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx512.mask.blend.w.256(<16 x i16>, <16 x i16>, i16) nounwind readonly

; CHECK-LABEL: test_x86_mask_blend_b_512
; CHECK: vpblendmb
define <64 x i8> @test_x86_mask_blend_b_512(i64 %a0, <64 x i8> %a1, <64 x i8> %a2) {
  %res = call <64 x i8> @llvm.x86.avx512.mask.blend.b.512(<64 x i8> %a1, <64 x i8> %a2, i64 %a0) ; <<64 x i8>> [#uses=1]
  ret <64 x i8> %res
}
declare <64 x i8> @llvm.x86.avx512.mask.blend.b.512(<64 x i8>, <64 x i8>, i64) nounwind readonly

; CHECK-LABEL: test_x86_mask_blend_w_512
define <32 x i16> @test_x86_mask_blend_w_512(i32 %mask, <32 x i16> %a1, <32 x i16> %a2) {
  ; CHECK: vpblendmw
  %res = call <32 x i16> @llvm.x86.avx512.mask.blend.w.512(<32 x i16> %a1, <32 x i16> %a2, i32 %mask) ; <<32 x i16>> [#uses=1]
  ret <32 x i16> %res
}
declare <32 x i16> @llvm.x86.avx512.mask.blend.w.512(<32 x i16>, <32 x i16>, i32) nounwind readonly

; CHECK-LABEL: test_x86_mask_blend_b_128
; CHECK: vpblendmb
define <16 x i8> @test_x86_mask_blend_b_128(i16 %a0, <16 x i8> %a1, <16 x i8> %a2) {
  %res = call <16 x i8> @llvm.x86.avx512.mask.blend.b.128(<16 x i8> %a1, <16 x i8> %a2, i16 %a0) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.avx512.mask.blend.b.128(<16 x i8>, <16 x i8>, i16) nounwind readonly

; CHECK-LABEL: test_x86_mask_blend_w_128
define <8 x i16> @test_x86_mask_blend_w_128(i8 %mask, <8 x i16> %a1, <8 x i16> %a2) {
  ; CHECK: vpblendmw
  %res = call <8 x i16> @llvm.x86.avx512.mask.blend.w.128(<8 x i16> %a1, <8 x i16> %a2, i8 %mask) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.avx512.mask.blend.w.128(<8 x i16>, <8 x i16>, i8) nounwind readonly

define <32 x i16> @test_mask_packs_epi32_rr_512(<16 x i32> %a, <16 x i32> %b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rr_512
  ;CHECK: vpackssdw       %zmm1, %zmm0, %zmm0  ## encoding: [0x62,0xf1,0x7d,0x48,0x6b,0xc1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packs_epi32_rrk_512(<16 x i32> %a, <16 x i32> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rrk_512
  ;CHECK: vpackssdw       %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0x6b,0xd1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packs_epi32_rrkz_512(<16 x i32> %a, <16 x i32> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rrkz_512
  ;CHECK: vpackssdw       %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0x6b,0xc1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packs_epi32_rm_512(<16 x i32> %a, <16 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rm_512
  ;CHECK: vpackssdw       (%rdi), %zmm0, %zmm0  ## encoding: [0x62,0xf1,0x7d,0x48,0x6b,0x07]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packs_epi32_rmk_512(<16 x i32> %a, <16 x i32>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmk_512
  ;CHECK: vpackssdw       (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0x6b,0x0f]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packs_epi32_rmkz_512(<16 x i32> %a, <16 x i32>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmkz_512
  ;CHECK: vpackssdw       (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0x6b,0x07]
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packs_epi32_rmb_512(<16 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmb_512
  ;CHECK: vpackssdw       (%rdi){1to16}, %zmm0, %zmm0  ## encoding: [0x62,0xf1,0x7d,0x58,0x6b,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packs_epi32_rmbk_512(<16 x i32> %a, i32* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmbk_512
  ;CHECK: vpackssdw       (%rdi){1to16}, %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x59,0x6b,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packs_epi32_rmbkz_512(<16 x i32> %a, i32* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmbkz_512
  ;CHECK: vpackssdw       (%rdi){1to16}, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xd9,0x6b,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.packssdw.512(<16 x i32>, <16 x i32>, <32 x i16>, i32)

define <64 x i8> @test_mask_packs_epi16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_packs_epi16_rr_512
  ;CHECK: vpacksswb       %zmm1, %zmm0, %zmm0  ## encoding: [0x62,0xf1,0xfd,0x48,0x63,0xc1]
  %res = call <64 x i8> @llvm.x86.avx512.mask.packsswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> zeroinitializer, i64 -1)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packs_epi16_rrk_512(<32 x i16> %a, <32 x i16> %b, <64 x i8> %passThru, i64 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rrk_512
  ;CHECK: vpacksswb       %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x49,0x63,0xd1]
  %res = call <64 x i8> @llvm.x86.avx512.mask.packsswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> %passThru, i64 %mask)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packs_epi16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i64 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rrkz_512
  ;CHECK: vpacksswb       %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xc9,0x63,0xc1]
  %res = call <64 x i8> @llvm.x86.avx512.mask.packsswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> zeroinitializer, i64 %mask)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packs_epi16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi16_rm_512
  ;CHECK: vpacksswb       (%rdi), %zmm0, %zmm0  ## encoding: [0x62,0xf1,0xfd,0x48,0x63,0x07]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <64 x i8> @llvm.x86.avx512.mask.packsswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> zeroinitializer, i64 -1)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packs_epi16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <64 x i8> %passThru, i64 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rmk_512
  ;CHECK: vpacksswb       (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x49,0x63,0x0f]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <64 x i8> @llvm.x86.avx512.mask.packsswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> %passThru, i64 %mask)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packs_epi16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i64 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rmkz_512
  ;CHECK: vpacksswb       (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xc9,0x63,0x07]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <64 x i8> @llvm.x86.avx512.mask.packsswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> zeroinitializer, i64 %mask)
  ret <64 x i8> %res
}

declare <64 x i8> @llvm.x86.avx512.mask.packsswb.512(<32 x i16>, <32 x i16>, <64 x i8>, i64)


define <32 x i16> @test_mask_packus_epi32_rr_512(<16 x i32> %a, <16 x i32> %b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rr_512
  ;CHECK: vpackusdw       %zmm1, %zmm0, %zmm0  
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packus_epi32_rrk_512(<16 x i32> %a, <16 x i32> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rrk_512
  ;CHECK: vpackusdw       %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packus_epi32_rrkz_512(<16 x i32> %a, <16 x i32> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rrkz_512
  ;CHECK: vpackusdw       %zmm1, %zmm0, %zmm0 {%k1} {z} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packus_epi32_rm_512(<16 x i32> %a, <16 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rm_512
  ;CHECK: vpackusdw       (%rdi), %zmm0, %zmm0  
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packus_epi32_rmk_512(<16 x i32> %a, <16 x i32>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmk_512
  ;CHECK: vpackusdw       (%rdi), %zmm0, %zmm1 {%k1} 
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packus_epi32_rmkz_512(<16 x i32> %a, <16 x i32>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmkz_512
  ;CHECK: vpackusdw       (%rdi), %zmm0, %zmm0 {%k1} {z} 
  %b = load <16 x i32>, <16 x i32>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packus_epi32_rmb_512(<16 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmb_512
  ;CHECK: vpackusdw       (%rdi){1to16}, %zmm0, %zmm0  
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packus_epi32_rmbk_512(<16 x i32> %a, i32* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmbk_512
  ;CHECK: vpackusdw       (%rdi){1to16}, %zmm0, %zmm1 {%k1} 
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_packus_epi32_rmbkz_512(<16 x i32> %a, i32* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmbkz_512
  ;CHECK: vpackusdw       (%rdi){1to16}, %zmm0, %zmm0 {%k1} {z} 
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <16 x i32> undef, i32 %q, i32 0
  %b = shufflevector <16 x i32> %vecinit.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %res = call <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32> %a, <16 x i32> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.packusdw.512(<16 x i32>, <16 x i32>, <32 x i16>, i32)

define <64 x i8> @test_mask_packus_epi16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_packus_epi16_rr_512
  ;CHECK: vpackuswb       %zmm1, %zmm0, %zmm0  
  %res = call <64 x i8> @llvm.x86.avx512.mask.packuswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> zeroinitializer, i64 -1)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packus_epi16_rrk_512(<32 x i16> %a, <32 x i16> %b, <64 x i8> %passThru, i64 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rrk_512
  ;CHECK: vpackuswb       %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <64 x i8> @llvm.x86.avx512.mask.packuswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> %passThru, i64 %mask)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packus_epi16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i64 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rrkz_512
  ;CHECK: vpackuswb       %zmm1, %zmm0, %zmm0 {%k1} {z} 
  %res = call <64 x i8> @llvm.x86.avx512.mask.packuswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> zeroinitializer, i64 %mask)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packus_epi16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi16_rm_512
  ;CHECK: vpackuswb       (%rdi), %zmm0, %zmm0  
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <64 x i8> @llvm.x86.avx512.mask.packuswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> zeroinitializer, i64 -1)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packus_epi16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <64 x i8> %passThru, i64 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rmk_512
  ;CHECK: vpackuswb       (%rdi), %zmm0, %zmm1 {%k1} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <64 x i8> @llvm.x86.avx512.mask.packuswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> %passThru, i64 %mask)
  ret <64 x i8> %res
}

define <64 x i8> @test_mask_packus_epi16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i64 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rmkz_512
  ;CHECK: vpackuswb       (%rdi), %zmm0, %zmm0 {%k1} {z} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <64 x i8> @llvm.x86.avx512.mask.packuswb.512(<32 x i16> %a, <32 x i16> %b, <64 x i8> zeroinitializer, i64 %mask)
  ret <64 x i8> %res
}

declare <64 x i8> @llvm.x86.avx512.mask.packuswb.512(<32 x i16>, <32 x i16>, <64 x i8>, i64)

define <32 x i16> @test_mask_adds_epi16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_adds_epi16_rr_512
  ;CHECK: vpaddsw %zmm1, %zmm0, %zmm0     
  %res = call <32 x i16> @llvm.x86.avx512.mask.padds.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epi16_rrk_512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rrk_512
  ;CHECK: vpaddsw %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.padds.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epi16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rrkz_512
  ;CHECK: vpaddsw %zmm1, %zmm0, %zmm0 {%k1} {z} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.padds.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epi16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epi16_rm_512
  ;CHECK: vpaddsw (%rdi), %zmm0, %zmm0    
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.padds.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epi16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rmk_512
  ;CHECK: vpaddsw (%rdi), %zmm0, %zmm1 {%k1} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.padds.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epi16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rmkz_512
  ;CHECK: vpaddsw (%rdi), %zmm0, %zmm0 {%k1} {z} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.padds.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.padds.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <32 x i16> @test_mask_subs_epi16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_subs_epi16_rr_512
  ;CHECK: vpsubsw %zmm1, %zmm0, %zmm0     
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubs.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epi16_rrk_512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rrk_512
  ;CHECK: vpsubsw %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubs.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epi16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rrkz_512
  ;CHECK: vpsubsw %zmm1, %zmm0, %zmm0 {%k1} {z} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubs.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epi16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epi16_rm_512
  ;CHECK: vpsubsw (%rdi), %zmm0, %zmm0    
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubs.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epi16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rmk_512
  ;CHECK: vpsubsw (%rdi), %zmm0, %zmm1 {%k1} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubs.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epi16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rmkz_512
  ;CHECK: vpsubsw (%rdi), %zmm0, %zmm0 {%k1} {z} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubs.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.psubs.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <32 x i16> @test_mask_adds_epu16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_adds_epu16_rr_512
  ;CHECK: vpaddusw %zmm1, %zmm0, %zmm0     
  %res = call <32 x i16> @llvm.x86.avx512.mask.paddus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epu16_rrk_512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rrk_512
  ;CHECK: vpaddusw %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.paddus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epu16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rrkz_512
  ;CHECK: vpaddusw %zmm1, %zmm0, %zmm0 {%k1} {z} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.paddus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epu16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epu16_rm_512
  ;CHECK: vpaddusw (%rdi), %zmm0, %zmm0    
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.paddus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epu16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rmk_512
  ;CHECK: vpaddusw (%rdi), %zmm0, %zmm1 {%k1} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.paddus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_adds_epu16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rmkz_512
  ;CHECK: vpaddusw (%rdi), %zmm0, %zmm0 {%k1} {z} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.paddus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.paddus.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <32 x i16> @test_mask_subs_epu16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_subs_epu16_rr_512
  ;CHECK: vpsubusw %zmm1, %zmm0, %zmm0     
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epu16_rrk_512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rrk_512
  ;CHECK: vpsubusw %zmm1, %zmm0, %zmm2 {%k1} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epu16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rrkz_512
  ;CHECK: vpsubusw %zmm1, %zmm0, %zmm0 {%k1} {z} 
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epu16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epu16_rm_512
  ;CHECK: vpsubusw (%rdi), %zmm0, %zmm0    
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epu16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rmk_512
  ;CHECK: vpsubusw (%rdi), %zmm0, %zmm1 {%k1} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_subs_epu16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rmkz_512
  ;CHECK: vpsubusw (%rdi), %zmm0, %zmm0 {%k1} {z} 
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psubus.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.psubus.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

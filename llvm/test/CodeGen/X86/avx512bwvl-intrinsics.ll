; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512bw -mattr=+avx512vl --show-mc-encoding| FileCheck %s

; 256-bit

define i32 @test_pcmpeq_b_256(<32 x i8> %a, <32 x i8> %b) {
; CHECK-LABEL: test_pcmpeq_b_256
; CHECK: vpcmpeqb %ymm1, %ymm0, %k0 ##
  %res = call i32 @llvm.x86.avx512.mask.pcmpeq.b.256(<32 x i8> %a, <32 x i8> %b, i32 -1)
  ret i32 %res
}

define i32 @test_mask_pcmpeq_b_256(<32 x i8> %a, <32 x i8> %b, i32 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_b_256
; CHECK: vpcmpeqb %ymm1, %ymm0, %k0 {%k1} ##
  %res = call i32 @llvm.x86.avx512.mask.pcmpeq.b.256(<32 x i8> %a, <32 x i8> %b, i32 %mask)
  ret i32 %res
}

declare i32 @llvm.x86.avx512.mask.pcmpeq.b.256(<32 x i8>, <32 x i8>, i32)

define i16 @test_pcmpeq_w_256(<16 x i16> %a, <16 x i16> %b) {
; CHECK-LABEL: test_pcmpeq_w_256
; CHECK: vpcmpeqw %ymm1, %ymm0, %k0 ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpeq.w.256(<16 x i16> %a, <16 x i16> %b, i16 -1)
  ret i16 %res
}

define i16 @test_mask_pcmpeq_w_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_w_256
; CHECK: vpcmpeqw %ymm1, %ymm0, %k0 {%k1} ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpeq.w.256(<16 x i16> %a, <16 x i16> %b, i16 %mask)
  ret i16 %res
}

declare i16 @llvm.x86.avx512.mask.pcmpeq.w.256(<16 x i16>, <16 x i16>, i16)

define i32 @test_pcmpgt_b_256(<32 x i8> %a, <32 x i8> %b) {
; CHECK-LABEL: test_pcmpgt_b_256
; CHECK: vpcmpgtb %ymm1, %ymm0, %k0 ##
  %res = call i32 @llvm.x86.avx512.mask.pcmpgt.b.256(<32 x i8> %a, <32 x i8> %b, i32 -1)
  ret i32 %res
}

define i32 @test_mask_pcmpgt_b_256(<32 x i8> %a, <32 x i8> %b, i32 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_b_256
; CHECK: vpcmpgtb %ymm1, %ymm0, %k0 {%k1} ##
  %res = call i32 @llvm.x86.avx512.mask.pcmpgt.b.256(<32 x i8> %a, <32 x i8> %b, i32 %mask)
  ret i32 %res
}

declare i32 @llvm.x86.avx512.mask.pcmpgt.b.256(<32 x i8>, <32 x i8>, i32)

define i16 @test_pcmpgt_w_256(<16 x i16> %a, <16 x i16> %b) {
; CHECK-LABEL: test_pcmpgt_w_256
; CHECK: vpcmpgtw %ymm1, %ymm0, %k0 ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpgt.w.256(<16 x i16> %a, <16 x i16> %b, i16 -1)
  ret i16 %res
}

define i16 @test_mask_pcmpgt_w_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_w_256
; CHECK: vpcmpgtw %ymm1, %ymm0, %k0 {%k1} ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpgt.w.256(<16 x i16> %a, <16 x i16> %b, i16 %mask)
  ret i16 %res
}

declare i16 @llvm.x86.avx512.mask.pcmpgt.w.256(<16 x i16>, <16 x i16>, i16)

define <8 x i32> @test_cmp_b_256(<32 x i8> %a0, <32 x i8> %a1) {
; CHECK_LABEL: test_cmp_b_256
; CHECK: vpcmpeqb %ymm1, %ymm0, %k0 ##
  %res0 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 0, i32 -1)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltb %ymm1, %ymm0, %k0 ##
  %res1 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 1, i32 -1)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmpleb %ymm1, %ymm0, %k0 ##
  %res2 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 2, i32 -1)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunordb %ymm1, %ymm0, %k0 ##
  %res3 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 3, i32 -1)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpneqb %ymm1, %ymm0, %k0 ##
  %res4 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 4, i32 -1)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltb %ymm1, %ymm0, %k0 ##
  %res5 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 5, i32 -1)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnleb %ymm1, %ymm0, %k0 ##
  %res6 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 6, i32 -1)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmpordb %ymm1, %ymm0, %k0 ##
  %res7 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 7, i32 -1)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

define <8 x i32> @test_mask_cmp_b_256(<32 x i8> %a0, <32 x i8> %a1, i32 %mask) {
; CHECK_LABEL: test_mask_cmp_b_256
; CHECK: vpcmpeqb %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 0, i32 %mask)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltb %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 1, i32 %mask)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmpleb %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 2, i32 %mask)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunordb %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 3, i32 %mask)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpneqb %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 4, i32 %mask)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltb %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 5, i32 %mask)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnleb %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 6, i32 %mask)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmpordb %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 7, i32 %mask)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

declare i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8>, <32 x i8>, i8, i32) nounwind readnone

define <8 x i32> @test_ucmp_b_256(<32 x i8> %a0, <32 x i8> %a1) {
; CHECK_LABEL: test_ucmp_b_256
; CHECK: vpcmpequb %ymm1, %ymm0, %k0 ##
  %res0 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 0, i32 -1)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltub %ymm1, %ymm0, %k0 ##
  %res1 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 1, i32 -1)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmpleub %ymm1, %ymm0, %k0 ##
  %res2 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 2, i32 -1)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunordub %ymm1, %ymm0, %k0 ##
  %res3 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 3, i32 -1)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpnequb %ymm1, %ymm0, %k0 ##
  %res4 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 4, i32 -1)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltub %ymm1, %ymm0, %k0 ##
  %res5 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 5, i32 -1)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnleub %ymm1, %ymm0, %k0 ##
  %res6 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 6, i32 -1)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmpordub %ymm1, %ymm0, %k0 ##
  %res7 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 7, i32 -1)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

define <8 x i32> @test_mask_ucmp_b_256(<32 x i8> %a0, <32 x i8> %a1, i32 %mask) {
; CHECK_LABEL: test_mask_ucmp_b_256
; CHECK: vpcmpequb %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 0, i32 %mask)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltub %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 1, i32 %mask)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmpleub %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 2, i32 %mask)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunordub %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 3, i32 %mask)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpnequb %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 4, i32 %mask)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltub %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 5, i32 %mask)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnleub %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 6, i32 %mask)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmpordub %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i8 7, i32 %mask)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

declare i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8>, <32 x i8>, i8, i32) nounwind readnone

define <8 x i16> @test_cmp_w_256(<16 x i16> %a0, <16 x i16> %a1) {
; CHECK_LABEL: test_cmp_w_256
; CHECK: vpcmpeqw %ymm1, %ymm0, %k0 ##
  %res0 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 0, i16 -1)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltw %ymm1, %ymm0, %k0 ##
  %res1 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 1, i16 -1)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmplew %ymm1, %ymm0, %k0 ##
  %res2 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 2, i16 -1)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordw %ymm1, %ymm0, %k0 ##
  %res3 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 3, i16 -1)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpneqw %ymm1, %ymm0, %k0 ##
  %res4 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 4, i16 -1)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltw %ymm1, %ymm0, %k0 ##
  %res5 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 5, i16 -1)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnlew %ymm1, %ymm0, %k0 ##
  %res6 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 6, i16 -1)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordw %ymm1, %ymm0, %k0 ##
  %res7 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 7, i16 -1)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

define <8 x i16> @test_mask_cmp_w_256(<16 x i16> %a0, <16 x i16> %a1, i16 %mask) {
; CHECK_LABEL: test_mask_cmp_w_256
; CHECK: vpcmpeqw %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 0, i16 %mask)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltw %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 1, i16 %mask)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmplew %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 2, i16 %mask)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordw %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 3, i16 %mask)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpneqw %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 4, i16 %mask)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltw %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 5, i16 %mask)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnlew %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 6, i16 %mask)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordw %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 7, i16 %mask)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

declare i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16>, <16 x i16>, i8, i16) nounwind readnone

define <8 x i16> @test_ucmp_w_256(<16 x i16> %a0, <16 x i16> %a1) {
; CHECK_LABEL: test_ucmp_w_256
; CHECK: vpcmpequw %ymm1, %ymm0, %k0 ##
  %res0 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 0, i16 -1)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltuw %ymm1, %ymm0, %k0 ##
  %res1 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 1, i16 -1)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleuw %ymm1, %ymm0, %k0 ##
  %res2 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 2, i16 -1)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunorduw %ymm1, %ymm0, %k0 ##
  %res3 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 3, i16 -1)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpnequw %ymm1, %ymm0, %k0 ##
  %res4 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 4, i16 -1)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltuw %ymm1, %ymm0, %k0 ##
  %res5 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 5, i16 -1)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleuw %ymm1, %ymm0, %k0 ##
  %res6 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 6, i16 -1)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmporduw %ymm1, %ymm0, %k0 ##
  %res7 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 7, i16 -1)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

define <8 x i16> @test_mask_ucmp_w_256(<16 x i16> %a0, <16 x i16> %a1, i16 %mask) {
; CHECK_LABEL: test_mask_ucmp_w_256
; CHECK: vpcmpequw %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 0, i16 %mask)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltuw %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 1, i16 %mask)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleuw %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 2, i16 %mask)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunorduw %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 3, i16 %mask)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpnequw %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 4, i16 %mask)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltuw %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 5, i16 %mask)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleuw %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 6, i16 %mask)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmporduw %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i8 7, i16 %mask)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

declare i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16>, <16 x i16>, i8, i16) nounwind readnone

; 128-bit

define i16 @test_pcmpeq_b_128(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_pcmpeq_b_128
; CHECK: vpcmpeqb %xmm1, %xmm0, %k0 ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpeq.b.128(<16 x i8> %a, <16 x i8> %b, i16 -1)
  ret i16 %res
}

define i16 @test_mask_pcmpeq_b_128(<16 x i8> %a, <16 x i8> %b, i16 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_b_128
; CHECK: vpcmpeqb %xmm1, %xmm0, %k0 {%k1} ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpeq.b.128(<16 x i8> %a, <16 x i8> %b, i16 %mask)
  ret i16 %res
}

declare i16 @llvm.x86.avx512.mask.pcmpeq.b.128(<16 x i8>, <16 x i8>, i16)

define i8 @test_pcmpeq_w_128(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_pcmpeq_w_128
; CHECK: vpcmpeqw %xmm1, %xmm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.w.128(<8 x i16> %a, <8 x i16> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpeq_w_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_w_128
; CHECK: vpcmpeqw %xmm1, %xmm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.w.128(<8 x i16> %a, <8 x i16> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpeq.w.128(<8 x i16>, <8 x i16>, i8)

define i16 @test_pcmpgt_b_128(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_pcmpgt_b_128
; CHECK: vpcmpgtb %xmm1, %xmm0, %k0 ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpgt.b.128(<16 x i8> %a, <16 x i8> %b, i16 -1)
  ret i16 %res
}

define i16 @test_mask_pcmpgt_b_128(<16 x i8> %a, <16 x i8> %b, i16 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_b_128
; CHECK: vpcmpgtb %xmm1, %xmm0, %k0 {%k1} ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpgt.b.128(<16 x i8> %a, <16 x i8> %b, i16 %mask)
  ret i16 %res
}

declare i16 @llvm.x86.avx512.mask.pcmpgt.b.128(<16 x i8>, <16 x i8>, i16)

define i8 @test_pcmpgt_w_128(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_pcmpgt_w_128
; CHECK: vpcmpgtw %xmm1, %xmm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.w.128(<8 x i16> %a, <8 x i16> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpgt_w_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_w_128
; CHECK: vpcmpgtw %xmm1, %xmm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.w.128(<8 x i16> %a, <8 x i16> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpgt.w.128(<8 x i16>, <8 x i16>, i8)

define <8 x i16> @test_cmp_b_128(<16 x i8> %a0, <16 x i8> %a1) {
; CHECK_LABEL: test_cmp_b_128
; CHECK: vpcmpeqb %xmm1, %xmm0, %k0 ##
  %res0 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 0, i16 -1)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltb %xmm1, %xmm0, %k0 ##
  %res1 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 1, i16 -1)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleb %xmm1, %xmm0, %k0 ##
  %res2 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 2, i16 -1)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordb %xmm1, %xmm0, %k0 ##
  %res3 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 3, i16 -1)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpneqb %xmm1, %xmm0, %k0 ##
  %res4 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 4, i16 -1)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltb %xmm1, %xmm0, %k0 ##
  %res5 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 5, i16 -1)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleb %xmm1, %xmm0, %k0 ##
  %res6 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 6, i16 -1)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordb %xmm1, %xmm0, %k0 ##
  %res7 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 7, i16 -1)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

define <8 x i16> @test_mask_cmp_b_128(<16 x i8> %a0, <16 x i8> %a1, i16 %mask) {
; CHECK_LABEL: test_mask_cmp_b_128
; CHECK: vpcmpeqb %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 0, i16 %mask)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltb %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 1, i16 %mask)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleb %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 2, i16 %mask)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordb %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 3, i16 %mask)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpneqb %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 4, i16 %mask)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltb %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 5, i16 %mask)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleb %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 6, i16 %mask)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordb %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 7, i16 %mask)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

declare i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8>, <16 x i8>, i8, i16) nounwind readnone

define <8 x i16> @test_ucmp_b_128(<16 x i8> %a0, <16 x i8> %a1) {
; CHECK_LABEL: test_ucmp_b_128
; CHECK: vpcmpequb %xmm1, %xmm0, %k0 ##
  %res0 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 0, i16 -1)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltub %xmm1, %xmm0, %k0 ##
  %res1 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 1, i16 -1)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleub %xmm1, %xmm0, %k0 ##
  %res2 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 2, i16 -1)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordub %xmm1, %xmm0, %k0 ##
  %res3 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 3, i16 -1)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpnequb %xmm1, %xmm0, %k0 ##
  %res4 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 4, i16 -1)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltub %xmm1, %xmm0, %k0 ##
  %res5 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 5, i16 -1)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleub %xmm1, %xmm0, %k0 ##
  %res6 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 6, i16 -1)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordub %xmm1, %xmm0, %k0 ##
  %res7 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 7, i16 -1)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

define <8 x i16> @test_mask_ucmp_b_128(<16 x i8> %a0, <16 x i8> %a1, i16 %mask) {
; CHECK_LABEL: test_mask_ucmp_b_128
; CHECK: vpcmpequb %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 0, i16 %mask)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltub %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 1, i16 %mask)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleub %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 2, i16 %mask)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordub %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 3, i16 %mask)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpnequb %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 4, i16 %mask)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltub %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 5, i16 %mask)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleub %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 6, i16 %mask)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordub %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i8 7, i16 %mask)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

declare i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8>, <16 x i8>, i8, i16) nounwind readnone

define <8 x i8> @test_cmp_w_128(<8 x i16> %a0, <8 x i16> %a1) {
; CHECK_LABEL: test_cmp_w_128
; CHECK: vpcmpeqw %xmm1, %xmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltw %xmm1, %xmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmplew %xmm1, %xmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordw %xmm1, %xmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqw %xmm1, %xmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltw %xmm1, %xmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnlew %xmm1, %xmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordw %xmm1, %xmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_cmp_w_128(<8 x i16> %a0, <8 x i16> %a1, i8 %mask) {
; CHECK_LABEL: test_mask_cmp_w_128
; CHECK: vpcmpeqw %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltw %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmplew %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordw %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqw %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltw %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnlew %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordw %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16>, <8 x i16>, i8, i8) nounwind readnone

define <8 x i8> @test_ucmp_w_128(<8 x i16> %a0, <8 x i16> %a1) {
; CHECK_LABEL: test_ucmp_w_128
; CHECK: vpcmpequw %xmm1, %xmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuw %xmm1, %xmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuw %xmm1, %xmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduw %xmm1, %xmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequw %xmm1, %xmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuw %xmm1, %xmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuw %xmm1, %xmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduw %xmm1, %xmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_ucmp_w_128(<8 x i16> %a0, <8 x i16> %a1, i8 %mask) {
; CHECK_LABEL: test_mask_ucmp_w_128
; CHECK: vpcmpequw %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuw %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuw %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduw %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequw %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuw %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuw %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduw %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i8 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16>, <8 x i16>, i8, i8) nounwind readnone

declare <8 x float> @llvm.x86.fma.mask.vfmadd.ps.256(<8 x float>, <8 x float>, <8 x float>, i8) nounwind readnone

define <8 x float> @test_mask_vfmadd256_ps(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd256_ps
  ; CHECK: vfmadd213ps %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x29,0xa8,0xc2]
  %res = call <8 x float> @llvm.x86.fma.mask.vfmadd.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 %mask) nounwind
  ret <8 x float> %res
}

declare <4 x float> @llvm.x86.fma.mask.vfmadd.ps.128(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone

define <4 x float> @test_mask_vfmadd128_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps
  ; CHECK: vfmadd213ps %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x09,0xa8,0xc2]
  %res = call <4 x float> @llvm.x86.fma.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) nounwind
  ret <4 x float> %res
}

declare <4 x double> @llvm.x86.fma.mask.vfmadd.pd.256(<4 x double>, <4 x double>, <4 x double>, i8)

define <4 x double> @test_mask_fmadd256_pd(<4 x double> %a, <4 x double> %b, <4 x double> %c, i8 %mask) {
; CHECK-LABEL: test_mask_fmadd256_pd:
; CHECK: vfmadd213pd %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x29,0xa8,0xc2]
  %res = call <4 x double> @llvm.x86.fma.mask.vfmadd.pd.256(<4 x double> %a, <4 x double> %b, <4 x double> %c, i8 %mask)
  ret <4 x double> %res
}

declare <2 x double> @llvm.x86.fma.mask.vfmadd.pd.128(<2 x double>, <2 x double>, <2 x double>, i8)

define <2 x double> @test_mask_fmadd128_pd(<2 x double> %a, <2 x double> %b, <2 x double> %c, i8 %mask) {
; CHECK-LABEL: test_mask_fmadd128_pd:
; CHECK: vfmadd213pd %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x09,0xa8,0xc2]
  %res = call <2 x double> @llvm.x86.fma.mask.vfmadd.pd.128(<2 x double> %a, <2 x double> %b, <2 x double> %c, i8 %mask)
  ret <2 x double> %res
}

declare <8 x float> @llvm.x86.fma.mask.vfmsub.ps.256(<8 x float>, <8 x float>, <8 x float>, i8) nounwind readnone

define <8 x float> @test_mask_vfmsub256_ps(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmsub256_ps
  ; CHECK: vfmsub213ps %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x29,0xaa,0xc2]
  %res = call <8 x float> @llvm.x86.fma.mask.vfmsub.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 %mask) nounwind
  ret <8 x float> %res
}

declare <4 x float> @llvm.x86.fma.mask.vfmsub.ps.128(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone

define <4 x float> @test_mask_vfmsub128_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmsub128_ps
  ; CHECK: vfmsub213ps %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x09,0xaa,0xc2]
  %res = call <4 x float> @llvm.x86.fma.mask.vfmsub.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) nounwind
  ret <4 x float> %res
}

declare <4 x double> @llvm.x86.fma.mask.vfmsub.pd.256(<4 x double>, <4 x double>, <4 x double>, i8) nounwind readnone

define <4 x double> @test_mask_vfmsub256_pd(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmsub256_pd
  ; CHECK: vfmsub213pd %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x29,0xaa,0xc2]
  %res = call <4 x double> @llvm.x86.fma.mask.vfmsub.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) nounwind
  ret <4 x double> %res
}

declare <2 x double> @llvm.x86.fma.mask.vfmsub.pd.128(<2 x double>, <2 x double>, <2 x double>, i8) nounwind readnone

define <2 x double> @test_mask_vfmsub128_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmsub128_pd
  ; CHECK: vfmsub213pd %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x09,0xaa,0xc2]
  %res = call <2 x double> @llvm.x86.fma.mask.vfmsub.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) nounwind
  ret <2 x double> %res
}

declare <8 x float> @llvm.x86.fma.mask.vfnmadd.ps.256(<8 x float>, <8 x float>, <8 x float>, i8) nounwind readnone

define <8 x float> @test_mask_vfnmadd256_ps(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmadd256_ps
  ; CHECK: vfnmadd213ps %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x29,0xac,0xc2]
  %res = call <8 x float> @llvm.x86.fma.mask.vfnmadd.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 %mask) nounwind
  ret <8 x float> %res
}

declare <4 x float> @llvm.x86.fma.mask.vfnmadd.ps.128(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone

define <4 x float> @test_mask_vfnmadd128_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmadd128_ps
  ; CHECK: vfnmadd213ps %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x09,0xac,0xc2]
  %res = call <4 x float> @llvm.x86.fma.mask.vfnmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) nounwind
  ret <4 x float> %res
}

declare <4 x double> @llvm.x86.fma.mask.vfnmadd.pd.256(<4 x double>, <4 x double>, <4 x double>, i8) nounwind readnone

define <4 x double> @test_mask_vfnmadd256_pd(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmadd256_pd
  ; CHECK: vfnmadd213pd %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x29,0xac,0xc2]
  %res = call <4 x double> @llvm.x86.fma.mask.vfnmadd.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) nounwind
  ret <4 x double> %res
}

declare <2 x double> @llvm.x86.fma.mask.vfnmadd.pd.128(<2 x double>, <2 x double>, <2 x double>, i8) nounwind readnone

define <2 x double> @test_mask_vfnmadd128_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmadd128_pd
  ; CHECK: vfnmadd213pd %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x09,0xac,0xc2]
  %res = call <2 x double> @llvm.x86.fma.mask.vfnmadd.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) nounwind
  ret <2 x double> %res
}

declare <8 x float> @llvm.x86.fma.mask.vfnmsub.ps.256(<8 x float>, <8 x float>, <8 x float>, i8) nounwind readnone

define <8 x float> @test_mask_vfnmsub256_ps(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmsub256_ps
  ; CHECK: vfnmsub213ps %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x29,0xae,0xc2]
  %res = call <8 x float> @llvm.x86.fma.mask.vfnmsub.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 %mask) nounwind
  ret <8 x float> %res
}

declare <4 x float> @llvm.x86.fma.mask.vfnmsub.ps.128(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone

define <4 x float> @test_mask_vfnmsub128_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmsub128_ps
  ; CHECK: vfnmsub213ps %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x09,0xae,0xc2]
  %res = call <4 x float> @llvm.x86.fma.mask.vfnmsub.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) nounwind
  ret <4 x float> %res
}

declare <4 x double> @llvm.x86.fma.mask.vfnmsub.pd.256(<4 x double>, <4 x double>, <4 x double>, i8) nounwind readnone

define <4 x double> @test_mask_vfnmsub256_pd(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmsub256_pd
  ; CHECK: vfnmsub213pd %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x29,0xae,0xc2]
  %res = call <4 x double> @llvm.x86.fma.mask.vfnmsub.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) nounwind
  ret <4 x double> %res
}

declare <2 x double> @llvm.x86.fma.mask.vfnmsub.pd.128(<2 x double>, <2 x double>, <2 x double>, i8) nounwind readnone

define <2 x double> @test_mask_vfnmsub128_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmsub128_pd
  ; CHECK: vfnmsub213pd %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x09,0xae,0xc2]
  %res = call <2 x double> @llvm.x86.fma.mask.vfnmsub.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) nounwind
  ret <2 x double> %res
}

declare <8 x float> @llvm.x86.fma.mask.vfmaddsub.ps.256(<8 x float>, <8 x float>, <8 x float>, i8) nounwind readnone

define <8 x float> @test_mask_fmaddsub256_ps(<8 x float> %a, <8 x float> %b, <8 x float> %c, i8 %mask) {
; CHECK-LABEL: test_mask_fmaddsub256_ps:
; CHECK: vfmaddsub213ps %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x29,0xa6,0xc2]
  %res = call <8 x float> @llvm.x86.fma.mask.vfmaddsub.ps.256(<8 x float> %a, <8 x float> %b, <8 x float> %c, i8 %mask)
  ret <8 x float> %res
}

declare <4 x float> @llvm.x86.fma.mask.vfmaddsub.ps.128(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone

define <4 x float> @test_mask_fmaddsub128_ps(<4 x float> %a, <4 x float> %b, <4 x float> %c, i8 %mask) {
; CHECK-LABEL: test_mask_fmaddsub128_ps:
; CHECK: vfmaddsub213ps %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x09,0xa6,0xc2]
  %res = call <4 x float> @llvm.x86.fma.mask.vfmaddsub.ps.128(<4 x float> %a, <4 x float> %b, <4 x float> %c, i8 %mask)
  ret <4 x float> %res
}

declare <4 x double> @llvm.x86.fma.mask.vfmaddsub.pd.256(<4 x double>, <4 x double>, <4 x double>, i8) nounwind readnone

define <4 x double> @test_mask_vfmaddsub256_pd(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmaddsub256_pd
  ; CHECK: vfmaddsub213pd %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x29,0xa6,0xc2]
  %res = call <4 x double> @llvm.x86.fma.mask.vfmaddsub.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) nounwind
  ret <4 x double> %res
}

declare <2 x double> @llvm.x86.fma.mask.vfmaddsub.pd.128(<2 x double>, <2 x double>, <2 x double>, i8) nounwind readnone

define <2 x double> @test_mask_vfmaddsub128_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmaddsub128_pd
  ; CHECK: vfmaddsub213pd %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x09,0xa6,0xc2]
  %res = call <2 x double> @llvm.x86.fma.mask.vfmaddsub.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) nounwind
  ret <2 x double> %res
}

declare <8 x float> @llvm.x86.fma.mask.vfmsubadd.ps.256(<8 x float>, <8 x float>, <8 x float>, i8) nounwind readnone

define <8 x float> @test_mask_vfmsubadd256_ps(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmsubadd256_ps
  ; CHECK: vfmsubadd213ps %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x29,0xa7,0xc2]
  %res = call <8 x float> @llvm.x86.fma.mask.vfmsubadd.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2, i8 %mask) nounwind
  ret <8 x float> %res
}

declare <4 x float> @llvm.x86.fma.mask.vfmsubadd.ps.128(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone

define <4 x float> @test_mask_vfmsubadd128_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmsubadd128_ps
  ; CHECK: vfmsubadd213ps %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x09,0xa7,0xc2]
  %res = call <4 x float> @llvm.x86.fma.mask.vfmsubadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) nounwind
  ret <4 x float> %res
}

declare <4 x double> @llvm.x86.fma.mask.vfmsubadd.pd.256(<4 x double>, <4 x double>, <4 x double>, i8) nounwind readnone

define <4 x double> @test_mask_vfmsubadd256_pd(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmsubadd256_pd
  ; CHECK: vfmsubadd213pd %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x29,0xa7,0xc2]
  %res = call <4 x double> @llvm.x86.fma.mask.vfmsubadd.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) nounwind
  ret <4 x double> %res
}
declare <2 x double> @llvm.x86.fma.mask.vfmsubadd.pd.128(<2 x double>, <2 x double>, <2 x double>, i8) nounwind readnone

define <2 x double> @test_mask_vfmsubadd128_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmsubadd128_pd
  ; CHECK: vfmsubadd213pd %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x09,0xa7,0xc2]
  %res = call <2 x double> @llvm.x86.fma.mask.vfmsubadd.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) nounwind
  ret <2 x double> %res
}

define <2 x double> @test_mask_vfmsubadd128rm_pd(<2 x double> %a0, <2 x double> %a1, <2 x double>* %ptr_a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmsubadd128rm_pd
  ; CHECK: vfmsubadd213pd (%rdi), %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x09,0xa7,0x07]
  %a2 = load <2 x double>, <2 x double>* %ptr_a2
  %res = call <2 x double> @llvm.x86.fma.mask.vfmsubadd.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) nounwind
  ret <2 x double> %res
}
declare <8 x double> @llvm.x86.fma.mask.vfmsubadd.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32) nounwind readnone
define <8 x double> @test_mask_vfmsubaddrm_pd(<8 x double> %a0, <8 x double> %a1, <8 x double>* %ptr_a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmsubaddrm_pd
  ; CHECK: vfmsubadd213pd  (%rdi), %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x49,0xa7,0x07]
  %a2 = load <8 x double>, <8 x double>* %ptr_a2, align 8
  %res = call <8 x double> @llvm.x86.fma.mask.vfmsubadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask, i32 4) nounwind
  ret <8 x double> %res
}

define <4 x float> @test_mask_vfmadd128_ps_r(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_r
  ; CHECK: vfmadd213ps %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x09,0xa8,0xc2]
  %res = call <4 x float> @llvm.x86.fma.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rz(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rz
  ; CHECK: vfmadd213ps %xmm2, %xmm1, %xmm0 ## encoding: [0x62,0xf2,0x75,0x08,0xa8,0xc2]
  %res = call <4 x float> @llvm.x86.fma.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 -1) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rmk(<4 x float> %a0, <4 x float> %a1, <4 x float>* %ptr_a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rmk
  ; CHECK: vfmadd213ps	(%rdi), %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x09,0xa8,0x07]
  %a2 = load <4 x float>, <4 x float>* %ptr_a2
  %res = call <4 x float> @llvm.x86.fma.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rmka(<4 x float> %a0, <4 x float> %a1, <4 x float>* %ptr_a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rmka
  ; CHECK: vfmadd213ps     (%rdi), %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x09,0xa8,0x07]
  %a2 = load <4 x float>, <4 x float>* %ptr_a2, align 8
  %res = call <4 x float> @llvm.x86.fma.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 %mask) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rmkz(<4 x float> %a0, <4 x float> %a1, <4 x float>* %ptr_a2) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rmkz
  ; CHECK: vfmadd213ps	(%rdi), %xmm1, %xmm0 ## encoding: [0xc4,0xe2,0x71,0xa8,0x07]
  %a2 = load <4 x float>, <4 x float>* %ptr_a2
  %res = call <4 x float> @llvm.x86.fma.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 -1) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rmkza(<4 x float> %a0, <4 x float> %a1, <4 x float>* %ptr_a2) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rmkza
  ; CHECK: vfmadd213ps	(%rdi), %xmm1, %xmm0 ## encoding: [0xc4,0xe2,0x71,0xa8,0x07]
  %a2 = load <4 x float>, <4 x float>* %ptr_a2, align 4
  %res = call <4 x float> @llvm.x86.fma.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2, i8 -1) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rmb(<4 x float> %a0, <4 x float> %a1, float* %ptr_a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rmb
  ; CHECK: vfmadd213ps	(%rdi){1to4}, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x19,0xa8,0x07]
  %q = load float, float* %ptr_a2
  %vecinit.i = insertelement <4 x float> undef, float %q, i32 0
  %vecinit2.i = insertelement <4 x float> %vecinit.i, float %q, i32 1
  %vecinit4.i = insertelement <4 x float> %vecinit2.i, float %q, i32 2
  %vecinit6.i = insertelement <4 x float> %vecinit4.i, float %q, i32 3
  %res = call <4 x float> @llvm.x86.fma.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %vecinit6.i, i8 %mask) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rmba(<4 x float> %a0, <4 x float> %a1, float* %ptr_a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rmba
  ; CHECK: vfmadd213ps	(%rdi){1to4}, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x19,0xa8,0x07]
  %q = load float, float* %ptr_a2, align 4
  %vecinit.i = insertelement <4 x float> undef, float %q, i32 0
  %vecinit2.i = insertelement <4 x float> %vecinit.i, float %q, i32 1
  %vecinit4.i = insertelement <4 x float> %vecinit2.i, float %q, i32 2
  %vecinit6.i = insertelement <4 x float> %vecinit4.i, float %q, i32 3
  %res = call <4 x float> @llvm.x86.fma.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %vecinit6.i, i8 %mask) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rmbz(<4 x float> %a0, <4 x float> %a1, float* %ptr_a2) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rmbz
  ; CHECK: vfmadd213ps	(%rdi){1to4}, %xmm1, %xmm0  ## encoding: [0x62,0xf2,0x75,0x18,0xa8,0x07]
  %q = load float, float* %ptr_a2
  %vecinit.i = insertelement <4 x float> undef, float %q, i32 0
  %vecinit2.i = insertelement <4 x float> %vecinit.i, float %q, i32 1
  %vecinit4.i = insertelement <4 x float> %vecinit2.i, float %q, i32 2
  %vecinit6.i = insertelement <4 x float> %vecinit4.i, float %q, i32 3
  %res = call <4 x float> @llvm.x86.fma.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %vecinit6.i, i8 -1) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_mask_vfmadd128_ps_rmbza(<4 x float> %a0, <4 x float> %a1, float* %ptr_a2) {
  ; CHECK-LABEL: test_mask_vfmadd128_ps_rmbza
  ; CHECK: vfmadd213ps	(%rdi){1to4}, %xmm1, %xmm0  ## encoding: [0x62,0xf2,0x75,0x18,0xa8,0x07]
  %q = load float, float* %ptr_a2, align 4
  %vecinit.i = insertelement <4 x float> undef, float %q, i32 0
  %vecinit2.i = insertelement <4 x float> %vecinit.i, float %q, i32 1
  %vecinit4.i = insertelement <4 x float> %vecinit2.i, float %q, i32 2
  %vecinit6.i = insertelement <4 x float> %vecinit4.i, float %q, i32 3
  %res = call <4 x float> @llvm.x86.fma.mask.vfmadd.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %vecinit6.i, i8 -1) nounwind
  ret <4 x float> %res
}

define <2 x double> @test_mask_vfmadd128_pd_r(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd128_pd_r
  ; CHECK: vfmadd213pd %xmm2, %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x09,0xa8,0xc2]
  %res = call <2 x double> @llvm.x86.fma.mask.vfmadd.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) nounwind
  ret <2 x double> %res
}

define <2 x double> @test_mask_vfmadd128_pd_rz(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK-LABEL: test_mask_vfmadd128_pd_rz
  ; CHECK: vfmadd213pd %xmm2, %xmm1, %xmm0 ## encoding: [0x62,0xf2,0xf5,0x08,0xa8,0xc2]
  %res = call <2 x double> @llvm.x86.fma.mask.vfmadd.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 -1) nounwind
  ret <2 x double> %res
}

define <2 x double> @test_mask_vfmadd128_pd_rmk(<2 x double> %a0, <2 x double> %a1, <2 x double>* %ptr_a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd128_pd_rmk
  ; CHECK: vfmadd213pd	(%rdi), %xmm1, %xmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x09,0xa8,0x07]
  %a2 = load <2 x double>, <2 x double>* %ptr_a2
  %res = call <2 x double> @llvm.x86.fma.mask.vfmadd.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 %mask) nounwind
  ret <2 x double> %res
}

define <2 x double> @test_mask_vfmadd128_pd_rmkz(<2 x double> %a0, <2 x double> %a1, <2 x double>* %ptr_a2) {
  ; CHECK-LABEL: test_mask_vfmadd128_pd_rmkz
  ; CHECK: vfmadd213pd	(%rdi), %xmm1, %xmm0 ## encoding: [0xc4,0xe2,0xf1,0xa8,0x07]
  %a2 = load <2 x double>, <2 x double>* %ptr_a2
  %res = call <2 x double> @llvm.x86.fma.mask.vfmadd.pd.128(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2, i8 -1) nounwind
  ret <2 x double> %res
}

define <4 x double> @test_mask_vfmadd256_pd_r(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd256_pd_r
  ; CHECK: vfmadd213pd %ymm2, %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x29,0xa8,0xc2]
  %res = call <4 x double> @llvm.x86.fma.mask.vfmadd.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) nounwind
  ret <4 x double> %res
}

define <4 x double> @test_mask_vfmadd256_pd_rz(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2) {
  ; CHECK-LABEL: test_mask_vfmadd256_pd_rz
  ; CHECK: vfmadd213pd %ymm2, %ymm1, %ymm0 ## encoding: [0x62,0xf2,0xf5,0x28,0xa8,0xc2]
  %res = call <4 x double> @llvm.x86.fma.mask.vfmadd.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 -1) nounwind
  ret <4 x double> %res
}

define <4 x double> @test_mask_vfmadd256_pd_rmk(<4 x double> %a0, <4 x double> %a1, <4 x double>* %ptr_a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmadd256_pd_rmk
  ; CHECK: vfmadd213pd	(%rdi), %ymm1, %ymm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x29,0xa8,0x07]
  %a2 = load <4 x double>, <4 x double>* %ptr_a2
  %res = call <4 x double> @llvm.x86.fma.mask.vfmadd.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 %mask) nounwind
  ret <4 x double> %res
}

define <4 x double> @test_mask_vfmadd256_pd_rmkz(<4 x double> %a0, <4 x double> %a1, <4 x double>* %ptr_a2) {
  ; CHECK-LABEL: test_mask_vfmadd256_pd_rmkz
  ; CHECK: vfmadd213pd	(%rdi), %ymm1, %ymm0 ## encoding: [0xc4,0xe2,0xf5,0xa8,0x07]
  %a2 = load <4 x double>, <4 x double>* %ptr_a2
  %res = call <4 x double> @llvm.x86.fma.mask.vfmadd.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2, i8 -1) nounwind
  ret <4 x double> %res
}

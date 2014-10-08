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
  %res0 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 0, i32 -1)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltb %ymm1, %ymm0, %k0 ##
  %res1 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 1, i32 -1)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmpleb %ymm1, %ymm0, %k0 ##
  %res2 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 2, i32 -1)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunordb %ymm1, %ymm0, %k0 ##
  %res3 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 3, i32 -1)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpneqb %ymm1, %ymm0, %k0 ##
  %res4 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 4, i32 -1)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltb %ymm1, %ymm0, %k0 ##
  %res5 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 5, i32 -1)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnleb %ymm1, %ymm0, %k0 ##
  %res6 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 6, i32 -1)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmpordb %ymm1, %ymm0, %k0 ##
  %res7 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 7, i32 -1)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

define <8 x i32> @test_mask_cmp_b_256(<32 x i8> %a0, <32 x i8> %a1, i32 %mask) {
; CHECK_LABEL: test_mask_cmp_b_256
; CHECK: vpcmpeqb %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 0, i32 %mask)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltb %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 1, i32 %mask)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmpleb %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 2, i32 %mask)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunordb %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 3, i32 %mask)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpneqb %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 4, i32 %mask)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltb %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 5, i32 %mask)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnleb %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 6, i32 %mask)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmpordb %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 7, i32 %mask)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

declare i32 @llvm.x86.avx512.mask.cmp.b.256(<32 x i8>, <32 x i8>, i32, i32) nounwind readnone

define <8 x i32> @test_ucmp_b_256(<32 x i8> %a0, <32 x i8> %a1) {
; CHECK_LABEL: test_ucmp_b_256
; CHECK: vpcmpequb %ymm1, %ymm0, %k0 ##
  %res0 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 0, i32 -1)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltub %ymm1, %ymm0, %k0 ##
  %res1 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 1, i32 -1)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmpleub %ymm1, %ymm0, %k0 ##
  %res2 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 2, i32 -1)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunordub %ymm1, %ymm0, %k0 ##
  %res3 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 3, i32 -1)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpnequb %ymm1, %ymm0, %k0 ##
  %res4 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 4, i32 -1)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltub %ymm1, %ymm0, %k0 ##
  %res5 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 5, i32 -1)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnleub %ymm1, %ymm0, %k0 ##
  %res6 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 6, i32 -1)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmpordub %ymm1, %ymm0, %k0 ##
  %res7 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 7, i32 -1)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

define <8 x i32> @test_mask_ucmp_b_256(<32 x i8> %a0, <32 x i8> %a1, i32 %mask) {
; CHECK_LABEL: test_mask_ucmp_b_256
; CHECK: vpcmpequb %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 0, i32 %mask)
  %vec0 = insertelement <8 x i32> undef, i32 %res0, i32 0
; CHECK: vpcmpltub %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 1, i32 %mask)
  %vec1 = insertelement <8 x i32> %vec0, i32 %res1, i32 1
; CHECK: vpcmpleub %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 2, i32 %mask)
  %vec2 = insertelement <8 x i32> %vec1, i32 %res2, i32 2
; CHECK: vpcmpunordub %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 3, i32 %mask)
  %vec3 = insertelement <8 x i32> %vec2, i32 %res3, i32 3
; CHECK: vpcmpnequb %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 4, i32 %mask)
  %vec4 = insertelement <8 x i32> %vec3, i32 %res4, i32 4
; CHECK: vpcmpnltub %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 5, i32 %mask)
  %vec5 = insertelement <8 x i32> %vec4, i32 %res5, i32 5
; CHECK: vpcmpnleub %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 6, i32 %mask)
  %vec6 = insertelement <8 x i32> %vec5, i32 %res6, i32 6
; CHECK: vpcmpordub %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> %a0, <32 x i8> %a1, i32 7, i32 %mask)
  %vec7 = insertelement <8 x i32> %vec6, i32 %res7, i32 7
  ret <8 x i32> %vec7
}

declare i32 @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8>, <32 x i8>, i32, i32) nounwind readnone

define <8 x i16> @test_cmp_w_256(<16 x i16> %a0, <16 x i16> %a1) {
; CHECK_LABEL: test_cmp_w_256
; CHECK: vpcmpeqw %ymm1, %ymm0, %k0 ##
  %res0 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 0, i16 -1)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltw %ymm1, %ymm0, %k0 ##
  %res1 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 1, i16 -1)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmplew %ymm1, %ymm0, %k0 ##
  %res2 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 2, i16 -1)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordw %ymm1, %ymm0, %k0 ##
  %res3 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 3, i16 -1)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpneqw %ymm1, %ymm0, %k0 ##
  %res4 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 4, i16 -1)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltw %ymm1, %ymm0, %k0 ##
  %res5 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 5, i16 -1)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnlew %ymm1, %ymm0, %k0 ##
  %res6 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 6, i16 -1)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordw %ymm1, %ymm0, %k0 ##
  %res7 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 7, i16 -1)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

define <8 x i16> @test_mask_cmp_w_256(<16 x i16> %a0, <16 x i16> %a1, i16 %mask) {
; CHECK_LABEL: test_mask_cmp_w_256
; CHECK: vpcmpeqw %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 0, i16 %mask)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltw %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 1, i16 %mask)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmplew %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 2, i16 %mask)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordw %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 3, i16 %mask)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpneqw %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 4, i16 %mask)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltw %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 5, i16 %mask)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnlew %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 6, i16 %mask)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordw %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 7, i16 %mask)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

declare i16 @llvm.x86.avx512.mask.cmp.w.256(<16 x i16>, <16 x i16>, i32, i16) nounwind readnone

define <8 x i16> @test_ucmp_w_256(<16 x i16> %a0, <16 x i16> %a1) {
; CHECK_LABEL: test_ucmp_w_256
; CHECK: vpcmpequw %ymm1, %ymm0, %k0 ##
  %res0 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 0, i16 -1)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltuw %ymm1, %ymm0, %k0 ##
  %res1 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 1, i16 -1)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleuw %ymm1, %ymm0, %k0 ##
  %res2 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 2, i16 -1)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunorduw %ymm1, %ymm0, %k0 ##
  %res3 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 3, i16 -1)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpnequw %ymm1, %ymm0, %k0 ##
  %res4 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 4, i16 -1)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltuw %ymm1, %ymm0, %k0 ##
  %res5 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 5, i16 -1)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleuw %ymm1, %ymm0, %k0 ##
  %res6 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 6, i16 -1)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmporduw %ymm1, %ymm0, %k0 ##
  %res7 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 7, i16 -1)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

define <8 x i16> @test_mask_ucmp_w_256(<16 x i16> %a0, <16 x i16> %a1, i16 %mask) {
; CHECK_LABEL: test_mask_ucmp_w_256
; CHECK: vpcmpequw %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 0, i16 %mask)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltuw %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 1, i16 %mask)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleuw %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 2, i16 %mask)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunorduw %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 3, i16 %mask)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpnequw %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 4, i16 %mask)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltuw %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 5, i16 %mask)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleuw %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 6, i16 %mask)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmporduw %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> %a0, <16 x i16> %a1, i32 7, i16 %mask)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

declare i16 @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16>, <16 x i16>, i32, i16) nounwind readnone

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
  %res0 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 0, i16 -1)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltb %xmm1, %xmm0, %k0 ##
  %res1 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 1, i16 -1)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleb %xmm1, %xmm0, %k0 ##
  %res2 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 2, i16 -1)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordb %xmm1, %xmm0, %k0 ##
  %res3 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 3, i16 -1)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpneqb %xmm1, %xmm0, %k0 ##
  %res4 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 4, i16 -1)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltb %xmm1, %xmm0, %k0 ##
  %res5 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 5, i16 -1)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleb %xmm1, %xmm0, %k0 ##
  %res6 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 6, i16 -1)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordb %xmm1, %xmm0, %k0 ##
  %res7 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 7, i16 -1)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

define <8 x i16> @test_mask_cmp_b_128(<16 x i8> %a0, <16 x i8> %a1, i16 %mask) {
; CHECK_LABEL: test_mask_cmp_b_128
; CHECK: vpcmpeqb %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 0, i16 %mask)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltb %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 1, i16 %mask)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleb %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 2, i16 %mask)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordb %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 3, i16 %mask)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpneqb %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 4, i16 %mask)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltb %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 5, i16 %mask)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleb %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 6, i16 %mask)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordb %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 7, i16 %mask)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

declare i16 @llvm.x86.avx512.mask.cmp.b.128(<16 x i8>, <16 x i8>, i32, i16) nounwind readnone

define <8 x i16> @test_ucmp_b_128(<16 x i8> %a0, <16 x i8> %a1) {
; CHECK_LABEL: test_ucmp_b_128
; CHECK: vpcmpequb %xmm1, %xmm0, %k0 ##
  %res0 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 0, i16 -1)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltub %xmm1, %xmm0, %k0 ##
  %res1 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 1, i16 -1)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleub %xmm1, %xmm0, %k0 ##
  %res2 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 2, i16 -1)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordub %xmm1, %xmm0, %k0 ##
  %res3 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 3, i16 -1)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpnequb %xmm1, %xmm0, %k0 ##
  %res4 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 4, i16 -1)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltub %xmm1, %xmm0, %k0 ##
  %res5 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 5, i16 -1)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleub %xmm1, %xmm0, %k0 ##
  %res6 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 6, i16 -1)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordub %xmm1, %xmm0, %k0 ##
  %res7 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 7, i16 -1)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

define <8 x i16> @test_mask_ucmp_b_128(<16 x i8> %a0, <16 x i8> %a1, i16 %mask) {
; CHECK_LABEL: test_mask_ucmp_b_128
; CHECK: vpcmpequb %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 0, i16 %mask)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltub %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 1, i16 %mask)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleub %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 2, i16 %mask)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordub %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 3, i16 %mask)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpnequb %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 4, i16 %mask)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltub %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 5, i16 %mask)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleub %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 6, i16 %mask)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordub %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> %a0, <16 x i8> %a1, i32 7, i16 %mask)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

declare i16 @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8>, <16 x i8>, i32, i16) nounwind readnone

define <8 x i8> @test_cmp_w_128(<8 x i16> %a0, <8 x i16> %a1) {
; CHECK_LABEL: test_cmp_w_128
; CHECK: vpcmpeqw %xmm1, %xmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltw %xmm1, %xmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmplew %xmm1, %xmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordw %xmm1, %xmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqw %xmm1, %xmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltw %xmm1, %xmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnlew %xmm1, %xmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordw %xmm1, %xmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_cmp_w_128(<8 x i16> %a0, <8 x i16> %a1, i8 %mask) {
; CHECK_LABEL: test_mask_cmp_w_128
; CHECK: vpcmpeqw %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltw %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmplew %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordw %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqw %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltw %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnlew %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordw %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.cmp.w.128(<8 x i16>, <8 x i16>, i32, i8) nounwind readnone

define <8 x i8> @test_ucmp_w_128(<8 x i16> %a0, <8 x i16> %a1) {
; CHECK_LABEL: test_ucmp_w_128
; CHECK: vpcmpequw %xmm1, %xmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuw %xmm1, %xmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuw %xmm1, %xmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduw %xmm1, %xmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequw %xmm1, %xmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuw %xmm1, %xmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuw %xmm1, %xmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduw %xmm1, %xmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_ucmp_w_128(<8 x i16> %a0, <8 x i16> %a1, i8 %mask) {
; CHECK_LABEL: test_mask_ucmp_w_128
; CHECK: vpcmpequw %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuw %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuw %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduw %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequw %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuw %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuw %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduw %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> %a0, <8 x i16> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16>, <8 x i16>, i32, i8) nounwind readnone

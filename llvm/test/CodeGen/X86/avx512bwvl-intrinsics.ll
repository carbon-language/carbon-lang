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
define <8 x i16> @test_mask_add_epi16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_add_epi16_rr_128
  ;CHECK: vpaddw %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xfd,0xc1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.padd.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_add_epi16_rrk_128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rrk_128
  ;CHECK: vpaddw %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xfd,0xd1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.padd.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_add_epi16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rrkz_128
  ;CHECK: vpaddw %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xfd,0xc1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.padd.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_add_epi16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi16_rm_128
  ;CHECK: vpaddw (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xfd,0x07]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.padd.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_add_epi16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rmk_128
  ;CHECK: vpaddw (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xfd,0x0f]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.padd.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_add_epi16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rmkz_128
  ;CHECK: vpaddw (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xfd,0x07]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.padd.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.padd.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i16> @test_mask_add_epi16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_add_epi16_rr_256
  ;CHECK: vpaddw %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xfd,0xc1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.padd.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_add_epi16_rrk_256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rrk_256
  ;CHECK: vpaddw %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xfd,0xd1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.padd.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_add_epi16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rrkz_256
  ;CHECK: vpaddw %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xfd,0xc1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.padd.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_add_epi16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi16_rm_256
  ;CHECK: vpaddw (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xfd,0x07]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.padd.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_add_epi16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rmk_256
  ;CHECK: vpaddw (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xfd,0x0f]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.padd.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_add_epi16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rmkz_256
  ;CHECK: vpaddw (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xfd,0x07]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.padd.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.padd.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <8 x i16> @test_mask_sub_epi16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_sub_epi16_rr_128
  ;CHECK: vpsubw %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xf9,0xc1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.psub.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_sub_epi16_rrk_128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rrk_128
  ;CHECK: vpsubw %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xf9,0xd1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.psub.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_sub_epi16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rrkz_128
  ;CHECK: vpsubw %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xf9,0xc1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.psub.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_sub_epi16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi16_rm_128
  ;CHECK: vpsubw (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xf9,0x07]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psub.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_sub_epi16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rmk_128
  ;CHECK: vpsubw (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xf9,0x0f]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psub.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_sub_epi16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rmkz_128
  ;CHECK: vpsubw (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xf9,0x07]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psub.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.psub.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i16> @test_mask_sub_epi16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_sub_epi16_rr_256
  ;CHECK: vpsubw %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xf9,0xc1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.psub.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_sub_epi16_rrk_256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rrk_256
  ;CHECK: vpsubw %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xf9,0xd1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.psub.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_sub_epi16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rrkz_256
  ;CHECK: vpsubw %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xf9,0xc1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.psub.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_sub_epi16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi16_rm_256
  ;CHECK: vpsubw (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xf9,0x07]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psub.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_sub_epi16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rmk_256
  ;CHECK: vpsubw (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xf9,0x0f]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psub.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_sub_epi16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rmkz_256
  ;CHECK: vpsubw (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xf9,0x07]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psub.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.psub.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <32 x i16> @test_mask_add_epi16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_add_epi16_rr_512
  ;CHECK: vpaddw %zmm1, %zmm0, %zmm0     ## encoding: [0x62,0xf1,0x7d,0x48,0xfd,0xc1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.padd.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_add_epi16_rrk_512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rrk_512
  ;CHECK: vpaddw %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xfd,0xd1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.padd.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_add_epi16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rrkz_512
  ;CHECK: vpaddw %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0xfd,0xc1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.padd.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_add_epi16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi16_rm_512
  ;CHECK: vpaddw (%rdi), %zmm0, %zmm0    ## encoding: [0x62,0xf1,0x7d,0x48,0xfd,0x07]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.padd.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_add_epi16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rmk_512
  ;CHECK: vpaddw (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xfd,0x0f]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.padd.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_add_epi16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_add_epi16_rmkz_512
  ;CHECK: vpaddw (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0xfd,0x07]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.padd.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.padd.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <32 x i16> @test_mask_sub_epi16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_sub_epi16_rr_512
  ;CHECK: vpsubw %zmm1, %zmm0, %zmm0     ## encoding: [0x62,0xf1,0x7d,0x48,0xf9,0xc1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.psub.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_sub_epi16_rrk_512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rrk_512
  ;CHECK: vpsubw %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xf9,0xd1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.psub.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_sub_epi16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rrkz_512
  ;CHECK: vpsubw %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0xf9,0xc1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.psub.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_sub_epi16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi16_rm_512
  ;CHECK: vpsubw (%rdi), %zmm0, %zmm0    ## encoding: [0x62,0xf1,0x7d,0x48,0xf9,0x07]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psub.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_sub_epi16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rmk_512
  ;CHECK: vpsubw (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xf9,0x0f]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psub.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_sub_epi16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi16_rmkz_512
  ;CHECK: vpsubw (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0xf9,0x07]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.psub.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.psub.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <32 x i16> @test_mask_mullo_epi16_rr_512(<32 x i16> %a, <32 x i16> %b) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rr_512
  ;CHECK: vpmullw %zmm1, %zmm0, %zmm0     ## encoding: [0x62,0xf1,0x7d,0x48,0xd5,0xc1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmull.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_mullo_epi16_rrk_512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rrk_512
  ;CHECK: vpmullw %zmm1, %zmm0, %zmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xd5,0xd1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmull.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_mullo_epi16_rrkz_512(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rrkz_512
  ;CHECK: vpmullw %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0xd5,0xc1]
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmull.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_mullo_epi16_rm_512(<32 x i16> %a, <32 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rm_512
  ;CHECK: vpmullw (%rdi), %zmm0, %zmm0    ## encoding: [0x62,0xf1,0x7d,0x48,0xd5,0x07]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmull.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 -1)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_mullo_epi16_rmk_512(<32 x i16> %a, <32 x i16>* %ptr_b, <32 x i16> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rmk_512
  ;CHECK: vpmullw (%rdi), %zmm0, %zmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x49,0xd5,0x0f]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmull.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> %passThru, i32 %mask)
  ret <32 x i16> %res
}

define <32 x i16> @test_mask_mullo_epi16_rmkz_512(<32 x i16> %a, <32 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rmkz_512
  ;CHECK: vpmullw (%rdi), %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xc9,0xd5,0x07]
  %b = load <32 x i16>, <32 x i16>* %ptr_b
  %res = call <32 x i16> @llvm.x86.avx512.mask.pmull.w.512(<32 x i16> %a, <32 x i16> %b, <32 x i16> zeroinitializer, i32 %mask)
  ret <32 x i16> %res
}

declare <32 x i16> @llvm.x86.avx512.mask.pmull.w.512(<32 x i16>, <32 x i16>, <32 x i16>, i32)

define <8 x i16> @test_mask_mullo_epi16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rr_128
  ;CHECK: vpmullw %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xd5,0xc1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmull.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_mullo_epi16_rrk_128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rrk_128
  ;CHECK: vpmullw %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xd5,0xd1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmull.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_mullo_epi16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rrkz_128
  ;CHECK: vpmullw %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xd5,0xc1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmull.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_mullo_epi16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rm_128
  ;CHECK: vpmullw (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xd5,0x07]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmull.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_mullo_epi16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rmk_128
  ;CHECK: vpmullw (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xd5,0x0f]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmull.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_mullo_epi16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rmkz_128
  ;CHECK: vpmullw (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xd5,0x07]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.pmull.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.pmull.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i16> @test_mask_mullo_epi16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rr_256
  ;CHECK: vpmullw %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xd5,0xc1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmull.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_mullo_epi16_rrk_256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rrk_256
  ;CHECK: vpmullw %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xd5,0xd1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmull.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_mullo_epi16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rrkz_256
  ;CHECK: vpmullw %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xd5,0xc1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmull.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_mullo_epi16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rm_256
  ;CHECK: vpmullw (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xd5,0x07]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmull.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_mullo_epi16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rmk_256
  ;CHECK: vpmullw (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xd5,0x0f]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmull.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_mullo_epi16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_mullo_epi16_rmkz_256
  ;CHECK: vpmullw (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xd5,0x07]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.pmull.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.pmull.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)


define <8 x i16> @test_mask_packs_epi32_rr_128(<4 x i32> %a, <4 x i32> %b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rr_128
  ;CHECK: vpackssdw       %xmm1, %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0x6b,0xc1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packs_epi32_rrk_128(<4 x i32> %a, <4 x i32> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rrk_128
  ;CHECK: vpackssdw       %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0x6b,0xd1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packs_epi32_rrkz_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rrkz_128
  ;CHECK: vpackssdw       %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0x6b,0xc1]
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packs_epi32_rm_128(<4 x i32> %a, <4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rm_128
  ;CHECK: vpackssdw       (%rdi), %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0x6b,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packs_epi32_rmk_128(<4 x i32> %a, <4 x i32>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmk_128
  ;CHECK: vpackssdw       (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0x6b,0x0f]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packs_epi32_rmkz_128(<4 x i32> %a, <4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmkz_128
  ;CHECK: vpackssdw       (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0x6b,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packs_epi32_rmb_128(<4 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmb_128
  ;CHECK: vpackssdw       (%rdi){1to4}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0x7d,0x18,0x6b,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packs_epi32_rmbk_128(<4 x i32> %a, i32* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmbk_128
  ;CHECK: vpackssdw       (%rdi){1to4}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x19,0x6b,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packs_epi32_rmbkz_128(<4 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmbkz_128
  ;CHECK: vpackssdw       (%rdi){1to4}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x99,0x6b,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.packssdw.128(<4 x i32>, <4 x i32>, <8 x i16>, i8)

define <16 x i16> @test_mask_packs_epi32_rr_256(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rr_256
  ;CHECK: vpackssdw       %ymm1, %ymm0, %ymm0 ## encoding: [0xc5,0xfd,0x6b,0xc1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packs_epi32_rrk_256(<8 x i32> %a, <8 x i32> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rrk_256
  ;CHECK: vpackssdw       %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0x6b,0xd1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packs_epi32_rrkz_256(<8 x i32> %a, <8 x i32> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rrkz_256
  ;CHECK: vpackssdw       %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0x6b,0xc1]
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packs_epi32_rm_256(<8 x i32> %a, <8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rm_256
  ;CHECK: vpackssdw       (%rdi), %ymm0, %ymm0 ## encoding: [0xc5,0xfd,0x6b,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packs_epi32_rmk_256(<8 x i32> %a, <8 x i32>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmk_256
  ;CHECK: vpackssdw       (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0x6b,0x0f]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packs_epi32_rmkz_256(<8 x i32> %a, <8 x i32>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmkz_256
  ;CHECK: vpackssdw       (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0x6b,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packs_epi32_rmb_256(<8 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmb_256
  ;CHECK: vpackssdw       (%rdi){1to8}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0x7d,0x38,0x6b,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packs_epi32_rmbk_256(<8 x i32> %a, i32* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmbk_256
  ;CHECK: vpackssdw       (%rdi){1to8}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x39,0x6b,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packs_epi32_rmbkz_256(<8 x i32> %a, i32* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi32_rmbkz_256
  ;CHECK: vpackssdw       (%rdi){1to8}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xb9,0x6b,0x07]  
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.packssdw.256(<8 x i32>, <8 x i32>, <16 x i16>, i16)

define <16 x i8> @test_mask_packs_epi16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_packs_epi16_rr_128
  ;CHECK: vpacksswb       %xmm1, %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0x63,0xc1]
  %res = call <16 x i8> @llvm.x86.avx512.mask.packsswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packs_epi16_rrk_128(<8 x i16> %a, <8 x i16> %b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rrk_128
  ;CHECK: vpacksswb       %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x09,0x63,0xd1]
  %res = call <16 x i8> @llvm.x86.avx512.mask.packsswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packs_epi16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rrkz_128
  ;CHECK: vpacksswb       %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x89,0x63,0xc1]
  %res = call <16 x i8> @llvm.x86.avx512.mask.packsswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packs_epi16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi16_rm_128
  ;CHECK: vpacksswb       (%rdi), %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0x63,0x07]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.packsswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packs_epi16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rmk_128
  ;CHECK: vpacksswb       (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x09,0x63,0x0f]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.packsswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packs_epi16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rmkz_128
  ;CHECK: vpacksswb       (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x89,0x63,0x07]
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.packsswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

declare <16 x i8> @llvm.x86.avx512.mask.packsswb.128(<8 x i16>, <8 x i16>, <16 x i8>, i16)

define <32 x i8> @test_mask_packs_epi16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_packs_epi16_rr_256
  ;CHECK: vpacksswb       %ymm1, %ymm0, %ymm0 ## encoding: [0xc5,0xfd,0x63,0xc1]
  %res = call <32 x i8> @llvm.x86.avx512.mask.packsswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packs_epi16_rrk_256(<16 x i16> %a, <16 x i16> %b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rrk_256
  ;CHECK: vpacksswb       %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x29,0x63,0xd1]
  %res = call <32 x i8> @llvm.x86.avx512.mask.packsswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packs_epi16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rrkz_256
  ;CHECK: vpacksswb       %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xa9,0x63,0xc1]
  %res = call <32 x i8> @llvm.x86.avx512.mask.packsswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packs_epi16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packs_epi16_rm_256
  ;CHECK: vpacksswb       (%rdi), %ymm0, %ymm0 ## encoding: [0xc5,0xfd,0x63,0x07]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.packsswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packs_epi16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rmk_256
  ;CHECK: vpacksswb       (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x29,0x63,0x0f]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.packsswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packs_epi16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packs_epi16_rmkz_256
  ;CHECK: vpacksswb       (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xa9,0x63,0x07]
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.packsswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

declare <32 x i8> @llvm.x86.avx512.mask.packsswb.256(<16 x i16>, <16 x i16>, <32 x i8>, i32)


define <8 x i16> @test_mask_packus_epi32_rr_128(<4 x i32> %a, <4 x i32> %b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rr_128
  ;CHECK: vpackusdw       %xmm1, %xmm0, %xmm0 
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packus_epi32_rrk_128(<4 x i32> %a, <4 x i32> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rrk_128
  ;CHECK: vpackusdw       %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packus_epi32_rrkz_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rrkz_128
  ;CHECK: vpackusdw       %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packus_epi32_rm_128(<4 x i32> %a, <4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rm_128
  ;CHECK: vpackusdw       (%rdi), %xmm0, %xmm0 
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packus_epi32_rmk_128(<4 x i32> %a, <4 x i32>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmk_128
  ;CHECK: vpackusdw       (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packus_epi32_rmkz_128(<4 x i32> %a, <4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmkz_128
  ;CHECK: vpackusdw       (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packus_epi32_rmb_128(<4 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmb_128
  ;CHECK: vpackusdw       (%rdi){1to4}, %xmm0, %xmm0  
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packus_epi32_rmbk_128(<4 x i32> %a, i32* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmbk_128
  ;CHECK: vpackusdw       (%rdi){1to4}, %xmm0, %xmm1 {%k1} 
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_packus_epi32_rmbkz_128(<4 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmbkz_128
  ;CHECK: vpackusdw       (%rdi){1to4}, %xmm0, %xmm0 {%k1} {z} 
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32> %a, <4 x i32> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.packusdw.128(<4 x i32>, <4 x i32>, <8 x i16>, i8)

define <16 x i16> @test_mask_packus_epi32_rr_256(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rr_256
  ;CHECK: vpackusdw       %ymm1, %ymm0, %ymm0 
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packus_epi32_rrk_256(<8 x i32> %a, <8 x i32> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rrk_256
  ;CHECK: vpackusdw       %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packus_epi32_rrkz_256(<8 x i32> %a, <8 x i32> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rrkz_256
  ;CHECK: vpackusdw       %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packus_epi32_rm_256(<8 x i32> %a, <8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rm_256
  ;CHECK: vpackusdw       (%rdi), %ymm0, %ymm0 
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packus_epi32_rmk_256(<8 x i32> %a, <8 x i32>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmk_256
  ;CHECK: vpackusdw       (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packus_epi32_rmkz_256(<8 x i32> %a, <8 x i32>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmkz_256
  ;CHECK: vpackusdw       (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packus_epi32_rmb_256(<8 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmb_256
  ;CHECK: vpackusdw       (%rdi){1to8}, %ymm0, %ymm0  
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packus_epi32_rmbk_256(<8 x i32> %a, i32* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmbk_256
  ;CHECK: vpackusdw       (%rdi){1to8}, %ymm0, %ymm1 {%k1} 
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_packus_epi32_rmbkz_256(<8 x i32> %a, i32* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi32_rmbkz_256
  ;CHECK: vpackusdw       (%rdi){1to8}, %ymm0, %ymm0 {%k1} {z} 
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32> %a, <8 x i32> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.packusdw.256(<8 x i32>, <8 x i32>, <16 x i16>, i16)

define <16 x i8> @test_mask_packus_epi16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_packus_epi16_rr_128
  ;CHECK: vpackuswb       %xmm1, %xmm0, %xmm0 
  %res = call <16 x i8> @llvm.x86.avx512.mask.packuswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packus_epi16_rrk_128(<8 x i16> %a, <8 x i16> %b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rrk_128
  ;CHECK: vpackuswb       %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.packuswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packus_epi16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rrkz_128
  ;CHECK: vpackuswb       %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.packuswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packus_epi16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi16_rm_128
  ;CHECK: vpackuswb       (%rdi), %xmm0, %xmm0 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.packuswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packus_epi16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rmk_128
  ;CHECK: vpackuswb       (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.packuswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_packus_epi16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rmkz_128
  ;CHECK: vpackuswb       (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.packuswb.128(<8 x i16> %a, <8 x i16> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

declare <16 x i8> @llvm.x86.avx512.mask.packuswb.128(<8 x i16>, <8 x i16>, <16 x i8>, i16)

define <32 x i8> @test_mask_packus_epi16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_packus_epi16_rr_256
  ;CHECK: vpackuswb       %ymm1, %ymm0, %ymm0 
  %res = call <32 x i8> @llvm.x86.avx512.mask.packuswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packus_epi16_rrk_256(<16 x i16> %a, <16 x i16> %b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rrk_256
  ;CHECK: vpackuswb       %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.packuswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packus_epi16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rrkz_256
  ;CHECK: vpackuswb       %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.packuswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packus_epi16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_packus_epi16_rm_256
  ;CHECK: vpackuswb       (%rdi), %ymm0, %ymm0 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.packuswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packus_epi16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rmk_256
  ;CHECK: vpackuswb       (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.packuswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_packus_epi16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_packus_epi16_rmkz_256
  ;CHECK: vpackuswb       (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.packuswb.256(<16 x i16> %a, <16 x i16> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

declare <32 x i8> @llvm.x86.avx512.mask.packuswb.256(<16 x i16>, <16 x i16>, <32 x i8>, i32)

define <8 x i16> @test_mask_adds_epi16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_adds_epi16_rr_128
  ;CHECK: vpaddsw %xmm1, %xmm0, %xmm0 
  %res = call <8 x i16> @llvm.x86.avx512.mask.padds.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epi16_rrk_128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rrk_128
  ;CHECK: vpaddsw %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.padds.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epi16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rrkz_128
  ;CHECK: vpaddsw %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.padds.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epi16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epi16_rm_128
  ;CHECK: vpaddsw (%rdi), %xmm0, %xmm0 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.padds.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epi16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rmk_128
  ;CHECK: vpaddsw (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.padds.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epi16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rmkz_128
  ;CHECK: vpaddsw (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.padds.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.padds.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i16> @test_mask_adds_epi16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_adds_epi16_rr_256
  ;CHECK: vpaddsw %ymm1, %ymm0, %ymm0 
  %res = call <16 x i16> @llvm.x86.avx512.mask.padds.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epi16_rrk_256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rrk_256
  ;CHECK: vpaddsw %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.padds.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epi16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rrkz_256
  ;CHECK: vpaddsw %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.padds.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epi16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epi16_rm_256
  ;CHECK: vpaddsw (%rdi), %ymm0, %ymm0    
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.padds.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epi16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rmk_256
  ;CHECK: vpaddsw (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.padds.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epi16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi16_rmkz_256
  ;CHECK: vpaddsw (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.padds.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.padds.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <8 x i16> @test_mask_subs_epi16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_subs_epi16_rr_128
  ;CHECK: vpsubsw %xmm1, %xmm0, %xmm0     
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubs.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epi16_rrk_128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rrk_128
  ;CHECK: vpsubsw %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubs.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epi16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rrkz_128
  ;CHECK: vpsubsw %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubs.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epi16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epi16_rm_128
  ;CHECK: vpsubsw (%rdi), %xmm0, %xmm0
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubs.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epi16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rmk_128
  ;CHECK: vpsubsw (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubs.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epi16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rmkz_128
  ;CHECK: vpsubsw (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubs.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.psubs.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i16> @test_mask_subs_epi16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_subs_epi16_rr_256
  ;CHECK: vpsubsw %ymm1, %ymm0, %ymm0     
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubs.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epi16_rrk_256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rrk_256
  ;CHECK: vpsubsw %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubs.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epi16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rrkz_256
  ;CHECK: vpsubsw %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubs.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epi16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epi16_rm_256
  ;CHECK: vpsubsw (%rdi), %ymm0, %ymm0    
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubs.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epi16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rmk_256
  ;CHECK: vpsubsw (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubs.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epi16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi16_rmkz_256
  ;CHECK: vpsubsw (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubs.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.psubs.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <8 x i16> @test_mask_adds_epu16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_adds_epu16_rr_128
  ;CHECK: vpaddusw %xmm1, %xmm0, %xmm0 
  %res = call <8 x i16> @llvm.x86.avx512.mask.paddus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epu16_rrk_128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rrk_128
  ;CHECK: vpaddusw %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.paddus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epu16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rrkz_128
  ;CHECK: vpaddusw %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.paddus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epu16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epu16_rm_128
  ;CHECK: vpaddusw (%rdi), %xmm0, %xmm0 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.paddus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epu16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rmk_128
  ;CHECK: vpaddusw (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.paddus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_adds_epu16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rmkz_128
  ;CHECK: vpaddusw (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.paddus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.paddus.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i16> @test_mask_adds_epu16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_adds_epu16_rr_256
  ;CHECK: vpaddusw %ymm1, %ymm0, %ymm0 
  %res = call <16 x i16> @llvm.x86.avx512.mask.paddus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epu16_rrk_256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rrk_256
  ;CHECK: vpaddusw %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.paddus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epu16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rrkz_256
  ;CHECK: vpaddusw %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.paddus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epu16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epu16_rm_256
  ;CHECK: vpaddusw (%rdi), %ymm0, %ymm0    
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.paddus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epu16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rmk_256
  ;CHECK: vpaddusw (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.paddus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_adds_epu16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu16_rmkz_256
  ;CHECK: vpaddusw (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.paddus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.paddus.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <8 x i16> @test_mask_subs_epu16_rr_128(<8 x i16> %a, <8 x i16> %b) {
  ;CHECK-LABEL: test_mask_subs_epu16_rr_128
  ;CHECK: vpsubusw %xmm1, %xmm0, %xmm0     
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epu16_rrk_128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rrk_128
  ;CHECK: vpsubusw %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epu16_rrkz_128(<8 x i16> %a, <8 x i16> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rrkz_128
  ;CHECK: vpsubusw %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epu16_rm_128(<8 x i16> %a, <8 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epu16_rm_128
  ;CHECK: vpsubusw (%rdi), %xmm0, %xmm0
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 -1)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epu16_rmk_128(<8 x i16> %a, <8 x i16>* %ptr_b, <8 x i16> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rmk_128
  ;CHECK: vpsubusw (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> %passThru, i8 %mask)
  ret <8 x i16> %res
}

define <8 x i16> @test_mask_subs_epu16_rmkz_128(<8 x i16> %a, <8 x i16>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rmkz_128
  ;CHECK: vpsubusw (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <8 x i16>, <8 x i16>* %ptr_b
  %res = call <8 x i16> @llvm.x86.avx512.mask.psubus.w.128(<8 x i16> %a, <8 x i16> %b, <8 x i16> zeroinitializer, i8 %mask)
  ret <8 x i16> %res
}

declare <8 x i16> @llvm.x86.avx512.mask.psubus.w.128(<8 x i16>, <8 x i16>, <8 x i16>, i8)

define <16 x i16> @test_mask_subs_epu16_rr_256(<16 x i16> %a, <16 x i16> %b) {
  ;CHECK-LABEL: test_mask_subs_epu16_rr_256
  ;CHECK: vpsubusw %ymm1, %ymm0, %ymm0     
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epu16_rrk_256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rrk_256
  ;CHECK: vpsubusw %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epu16_rrkz_256(<16 x i16> %a, <16 x i16> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rrkz_256
  ;CHECK: vpsubusw %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epu16_rm_256(<16 x i16> %a, <16 x i16>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epu16_rm_256
  ;CHECK: vpsubusw (%rdi), %ymm0, %ymm0    
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epu16_rmk_256(<16 x i16> %a, <16 x i16>* %ptr_b, <16 x i16> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rmk_256
  ;CHECK: vpsubusw (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> %passThru, i16 %mask)
  ret <16 x i16> %res
}

define <16 x i16> @test_mask_subs_epu16_rmkz_256(<16 x i16> %a, <16 x i16>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu16_rmkz_256
  ;CHECK: vpsubusw (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <16 x i16>, <16 x i16>* %ptr_b
  %res = call <16 x i16> @llvm.x86.avx512.mask.psubus.w.256(<16 x i16> %a, <16 x i16> %b, <16 x i16> zeroinitializer, i16 %mask)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.psubus.w.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)

define <16 x i8> @test_mask_adds_epi8_rr_128(<16 x i8> %a, <16 x i8> %b) {
  ;CHECK-LABEL: test_mask_adds_epi8_rr_128
  ;CHECK: vpaddsb %xmm1, %xmm0, %xmm0 
  %res = call <16 x i8> @llvm.x86.avx512.mask.padds.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epi8_rrk_128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi8_rrk_128
  ;CHECK: vpaddsb %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.padds.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epi8_rrkz_128(<16 x i8> %a, <16 x i8> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi8_rrkz_128
  ;CHECK: vpaddsb %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.padds.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epi8_rm_128(<16 x i8> %a, <16 x i8>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epi8_rm_128
  ;CHECK: vpaddsb (%rdi), %xmm0, %xmm0 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.padds.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epi8_rmk_128(<16 x i8> %a, <16 x i8>* %ptr_b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi8_rmk_128
  ;CHECK: vpaddsb (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.padds.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epi8_rmkz_128(<16 x i8> %a, <16 x i8>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi8_rmkz_128
  ;CHECK: vpaddsb (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.padds.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

declare <16 x i8> @llvm.x86.avx512.mask.padds.b.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

define <32 x i8> @test_mask_adds_epi8_rr_256(<32 x i8> %a, <32 x i8> %b) {
  ;CHECK-LABEL: test_mask_adds_epi8_rr_256
  ;CHECK: vpaddsb %ymm1, %ymm0, %ymm0 
  %res = call <32 x i8> @llvm.x86.avx512.mask.padds.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epi8_rrk_256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi8_rrk_256
  ;CHECK: vpaddsb %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.padds.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epi8_rrkz_256(<32 x i8> %a, <32 x i8> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi8_rrkz_256
  ;CHECK: vpaddsb %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.padds.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epi8_rm_256(<32 x i8> %a, <32 x i8>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epi8_rm_256
  ;CHECK: vpaddsb (%rdi), %ymm0, %ymm0    
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.padds.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epi8_rmk_256(<32 x i8> %a, <32 x i8>* %ptr_b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi8_rmk_256
  ;CHECK: vpaddsb (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.padds.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epi8_rmkz_256(<32 x i8> %a, <32 x i8>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epi8_rmkz_256
  ;CHECK: vpaddsb (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.padds.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

declare <32 x i8> @llvm.x86.avx512.mask.padds.b.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

define <16 x i8> @test_mask_subs_epi8_rr_128(<16 x i8> %a, <16 x i8> %b) {
  ;CHECK-LABEL: test_mask_subs_epi8_rr_128
  ;CHECK: vpsubsb %xmm1, %xmm0, %xmm0     
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubs.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epi8_rrk_128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi8_rrk_128
  ;CHECK: vpsubsb %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubs.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epi8_rrkz_128(<16 x i8> %a, <16 x i8> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi8_rrkz_128
  ;CHECK: vpsubsb %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubs.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epi8_rm_128(<16 x i8> %a, <16 x i8>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epi8_rm_128
  ;CHECK: vpsubsb (%rdi), %xmm0, %xmm0
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubs.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epi8_rmk_128(<16 x i8> %a, <16 x i8>* %ptr_b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi8_rmk_128
  ;CHECK: vpsubsb (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubs.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epi8_rmkz_128(<16 x i8> %a, <16 x i8>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi8_rmkz_128
  ;CHECK: vpsubsb (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubs.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

declare <16 x i8> @llvm.x86.avx512.mask.psubs.b.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

define <32 x i8> @test_mask_subs_epi8_rr_256(<32 x i8> %a, <32 x i8> %b) {
  ;CHECK-LABEL: test_mask_subs_epi8_rr_256
  ;CHECK: vpsubsb %ymm1, %ymm0, %ymm0     
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubs.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epi8_rrk_256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi8_rrk_256
  ;CHECK: vpsubsb %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubs.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epi8_rrkz_256(<32 x i8> %a, <32 x i8> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi8_rrkz_256
  ;CHECK: vpsubsb %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubs.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epi8_rm_256(<32 x i8> %a, <32 x i8>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epi8_rm_256
  ;CHECK: vpsubsb (%rdi), %ymm0, %ymm0    
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubs.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epi8_rmk_256(<32 x i8> %a, <32 x i8>* %ptr_b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi8_rmk_256
  ;CHECK: vpsubsb (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubs.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epi8_rmkz_256(<32 x i8> %a, <32 x i8>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epi8_rmkz_256
  ;CHECK: vpsubsb (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubs.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

declare <32 x i8> @llvm.x86.avx512.mask.psubs.b.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

define <16 x i8> @test_mask_adds_epu8_rr_128(<16 x i8> %a, <16 x i8> %b) {
  ;CHECK-LABEL: test_mask_adds_epu8_rr_128
  ;CHECK: vpaddusb %xmm1, %xmm0, %xmm0 
  %res = call <16 x i8> @llvm.x86.avx512.mask.paddus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epu8_rrk_128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu8_rrk_128
  ;CHECK: vpaddusb %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.paddus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epu8_rrkz_128(<16 x i8> %a, <16 x i8> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu8_rrkz_128
  ;CHECK: vpaddusb %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.paddus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epu8_rm_128(<16 x i8> %a, <16 x i8>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epu8_rm_128
  ;CHECK: vpaddusb (%rdi), %xmm0, %xmm0 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.paddus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epu8_rmk_128(<16 x i8> %a, <16 x i8>* %ptr_b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu8_rmk_128
  ;CHECK: vpaddusb (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.paddus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_adds_epu8_rmkz_128(<16 x i8> %a, <16 x i8>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu8_rmkz_128
  ;CHECK: vpaddusb (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.paddus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

declare <16 x i8> @llvm.x86.avx512.mask.paddus.b.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

define <32 x i8> @test_mask_adds_epu8_rr_256(<32 x i8> %a, <32 x i8> %b) {
  ;CHECK-LABEL: test_mask_adds_epu8_rr_256
  ;CHECK: vpaddusb %ymm1, %ymm0, %ymm0 
  %res = call <32 x i8> @llvm.x86.avx512.mask.paddus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epu8_rrk_256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu8_rrk_256
  ;CHECK: vpaddusb %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.paddus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epu8_rrkz_256(<32 x i8> %a, <32 x i8> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu8_rrkz_256
  ;CHECK: vpaddusb %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.paddus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epu8_rm_256(<32 x i8> %a, <32 x i8>* %ptr_b) {
  ;CHECK-LABEL: test_mask_adds_epu8_rm_256
  ;CHECK: vpaddusb (%rdi), %ymm0, %ymm0    
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.paddus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epu8_rmk_256(<32 x i8> %a, <32 x i8>* %ptr_b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu8_rmk_256
  ;CHECK: vpaddusb (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.paddus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_adds_epu8_rmkz_256(<32 x i8> %a, <32 x i8>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_adds_epu8_rmkz_256
  ;CHECK: vpaddusb (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.paddus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

declare <32 x i8> @llvm.x86.avx512.mask.paddus.b.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)

define <16 x i8> @test_mask_subs_epu8_rr_128(<16 x i8> %a, <16 x i8> %b) {
  ;CHECK-LABEL: test_mask_subs_epu8_rr_128
  ;CHECK: vpsubusb %xmm1, %xmm0, %xmm0     
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epu8_rrk_128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu8_rrk_128
  ;CHECK: vpsubusb %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epu8_rrkz_128(<16 x i8> %a, <16 x i8> %b, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu8_rrkz_128
  ;CHECK: vpsubusb %xmm1, %xmm0, %xmm0 {%k1} {z} 
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epu8_rm_128(<16 x i8> %a, <16 x i8>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epu8_rm_128
  ;CHECK: vpsubusb (%rdi), %xmm0, %xmm0
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 -1)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epu8_rmk_128(<16 x i8> %a, <16 x i8>* %ptr_b, <16 x i8> %passThru, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu8_rmk_128
  ;CHECK: vpsubusb (%rdi), %xmm0, %xmm1 {%k1} 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> %passThru, i16 %mask)
  ret <16 x i8> %res
}

define <16 x i8> @test_mask_subs_epu8_rmkz_128(<16 x i8> %a, <16 x i8>* %ptr_b, i16 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu8_rmkz_128
  ;CHECK: vpsubusb (%rdi), %xmm0, %xmm0 {%k1} {z} 
  %b = load <16 x i8>, <16 x i8>* %ptr_b
  %res = call <16 x i8> @llvm.x86.avx512.mask.psubus.b.128(<16 x i8> %a, <16 x i8> %b, <16 x i8> zeroinitializer, i16 %mask)
  ret <16 x i8> %res
}

declare <16 x i8> @llvm.x86.avx512.mask.psubus.b.128(<16 x i8>, <16 x i8>, <16 x i8>, i16)

define <32 x i8> @test_mask_subs_epu8_rr_256(<32 x i8> %a, <32 x i8> %b) {
  ;CHECK-LABEL: test_mask_subs_epu8_rr_256
  ;CHECK: vpsubusb %ymm1, %ymm0, %ymm0     
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epu8_rrk_256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu8_rrk_256
  ;CHECK: vpsubusb %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epu8_rrkz_256(<32 x i8> %a, <32 x i8> %b, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu8_rrkz_256
  ;CHECK: vpsubusb %ymm1, %ymm0, %ymm0 {%k1} {z} 
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epu8_rm_256(<32 x i8> %a, <32 x i8>* %ptr_b) {
  ;CHECK-LABEL: test_mask_subs_epu8_rm_256
  ;CHECK: vpsubusb (%rdi), %ymm0, %ymm0    
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epu8_rmk_256(<32 x i8> %a, <32 x i8>* %ptr_b, <32 x i8> %passThru, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu8_rmk_256
  ;CHECK: vpsubusb (%rdi), %ymm0, %ymm1 {%k1} 
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> %passThru, i32 %mask)
  ret <32 x i8> %res
}

define <32 x i8> @test_mask_subs_epu8_rmkz_256(<32 x i8> %a, <32 x i8>* %ptr_b, i32 %mask) {
  ;CHECK-LABEL: test_mask_subs_epu8_rmkz_256
  ;CHECK: vpsubusb (%rdi), %ymm0, %ymm0 {%k1} {z} 
  %b = load <32 x i8>, <32 x i8>* %ptr_b
  %res = call <32 x i8> @llvm.x86.avx512.mask.psubus.b.256(<32 x i8> %a, <32 x i8> %b, <32 x i8> zeroinitializer, i32 %mask)
  ret <32 x i8> %res
}

declare <32 x i8> @llvm.x86.avx512.mask.psubus.b.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)
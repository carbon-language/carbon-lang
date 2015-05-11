; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512vl --show-mc-encoding| FileCheck %s

; 256-bit

define i8 @test_pcmpeq_d_256(<8 x i32> %a, <8 x i32> %b) {
; CHECK-LABEL: test_pcmpeq_d_256
; CHECK: vpcmpeqd %ymm1, %ymm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.d.256(<8 x i32> %a, <8 x i32> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpeq_d_256(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_d_256
; CHECK: vpcmpeqd %ymm1, %ymm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.d.256(<8 x i32> %a, <8 x i32> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpeq.d.256(<8 x i32>, <8 x i32>, i8)

define i8 @test_pcmpeq_q_256(<4 x i64> %a, <4 x i64> %b) {
; CHECK-LABEL: test_pcmpeq_q_256
; CHECK: vpcmpeqq %ymm1, %ymm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.q.256(<4 x i64> %a, <4 x i64> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpeq_q_256(<4 x i64> %a, <4 x i64> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_q_256
; CHECK: vpcmpeqq %ymm1, %ymm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.q.256(<4 x i64> %a, <4 x i64> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpeq.q.256(<4 x i64>, <4 x i64>, i8)

define i8 @test_pcmpgt_d_256(<8 x i32> %a, <8 x i32> %b) {
; CHECK-LABEL: test_pcmpgt_d_256
; CHECK: vpcmpgtd %ymm1, %ymm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.d.256(<8 x i32> %a, <8 x i32> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpgt_d_256(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_d_256
; CHECK: vpcmpgtd %ymm1, %ymm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.d.256(<8 x i32> %a, <8 x i32> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpgt.d.256(<8 x i32>, <8 x i32>, i8)

define i8 @test_pcmpgt_q_256(<4 x i64> %a, <4 x i64> %b) {
; CHECK-LABEL: test_pcmpgt_q_256
; CHECK: vpcmpgtq %ymm1, %ymm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.q.256(<4 x i64> %a, <4 x i64> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpgt_q_256(<4 x i64> %a, <4 x i64> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_q_256
; CHECK: vpcmpgtq %ymm1, %ymm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.q.256(<4 x i64> %a, <4 x i64> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpgt.q.256(<4 x i64>, <4 x i64>, i8)

define <8 x i8> @test_cmp_d_256(<8 x i32> %a0, <8 x i32> %a1) {
; CHECK-LABEL: test_cmp_d_256
; CHECK: vpcmpeqd %ymm1, %ymm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltd %ymm1, %ymm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpled %ymm1, %ymm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordd %ymm1, %ymm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqd %ymm1, %ymm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltd %ymm1, %ymm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnled %ymm1, %ymm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordd %ymm1, %ymm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_cmp_d_256(<8 x i32> %a0, <8 x i32> %a1, i8 %mask) {
; CHECK-LABEL: test_mask_cmp_d_256
; CHECK: vpcmpeqd %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltd %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpled %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordd %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqd %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltd %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnled %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordd %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.cmp.d.256(<8 x i32>, <8 x i32>, i32, i8) nounwind readnone

define <8 x i8> @test_ucmp_d_256(<8 x i32> %a0, <8 x i32> %a1) {
; CHECK-LABEL: test_ucmp_d_256
; CHECK: vpcmpequd %ymm1, %ymm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltud %ymm1, %ymm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleud %ymm1, %ymm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordud %ymm1, %ymm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequd %ymm1, %ymm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltud %ymm1, %ymm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleud %ymm1, %ymm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordud %ymm1, %ymm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_ucmp_d_256(<8 x i32> %a0, <8 x i32> %a1, i8 %mask) {
; CHECK-LABEL: test_mask_ucmp_d_256
; CHECK: vpcmpequd %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltud %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleud %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordud %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequd %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltud %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleud %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordud %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> %a0, <8 x i32> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32>, <8 x i32>, i32, i8) nounwind readnone

define <8 x i8> @test_cmp_q_256(<4 x i64> %a0, <4 x i64> %a1) {
; CHECK-LABEL: test_cmp_q_256
; CHECK: vpcmpeqq %ymm1, %ymm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltq %ymm1, %ymm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleq %ymm1, %ymm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordq %ymm1, %ymm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqq %ymm1, %ymm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltq %ymm1, %ymm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleq %ymm1, %ymm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordq %ymm1, %ymm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_cmp_q_256(<4 x i64> %a0, <4 x i64> %a1, i8 %mask) {
; CHECK-LABEL: test_mask_cmp_q_256
; CHECK: vpcmpeqq %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltq %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleq %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordq %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqq %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltq %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleq %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordq %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.cmp.q.256(<4 x i64>, <4 x i64>, i32, i8) nounwind readnone

define <8 x i8> @test_ucmp_q_256(<4 x i64> %a0, <4 x i64> %a1) {
; CHECK-LABEL: test_ucmp_q_256
; CHECK: vpcmpequq %ymm1, %ymm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuq %ymm1, %ymm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuq %ymm1, %ymm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduq %ymm1, %ymm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequq %ymm1, %ymm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuq %ymm1, %ymm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuq %ymm1, %ymm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduq %ymm1, %ymm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_ucmp_q_256(<4 x i64> %a0, <4 x i64> %a1, i8 %mask) {
; CHECK-LABEL: test_mask_ucmp_q_256
; CHECK: vpcmpequq %ymm1, %ymm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuq %ymm1, %ymm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuq %ymm1, %ymm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduq %ymm1, %ymm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequq %ymm1, %ymm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuq %ymm1, %ymm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuq %ymm1, %ymm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduq %ymm1, %ymm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> %a0, <4 x i64> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64>, <4 x i64>, i32, i8) nounwind readnone

; 128-bit

define i8 @test_pcmpeq_d_128(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_pcmpeq_d_128
; CHECK: vpcmpeqd %xmm1, %xmm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.d.128(<4 x i32> %a, <4 x i32> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpeq_d_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_d_128
; CHECK: vpcmpeqd %xmm1, %xmm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.d.128(<4 x i32> %a, <4 x i32> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpeq.d.128(<4 x i32>, <4 x i32>, i8)

define i8 @test_pcmpeq_q_128(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_pcmpeq_q_128
; CHECK: vpcmpeqq %xmm1, %xmm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.q.128(<2 x i64> %a, <2 x i64> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpeq_q_128(<2 x i64> %a, <2 x i64> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_q_128
; CHECK: vpcmpeqq %xmm1, %xmm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.q.128(<2 x i64> %a, <2 x i64> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpeq.q.128(<2 x i64>, <2 x i64>, i8)

define i8 @test_pcmpgt_d_128(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_pcmpgt_d_128
; CHECK: vpcmpgtd %xmm1, %xmm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.d.128(<4 x i32> %a, <4 x i32> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpgt_d_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_d_128
; CHECK: vpcmpgtd %xmm1, %xmm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.d.128(<4 x i32> %a, <4 x i32> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpgt.d.128(<4 x i32>, <4 x i32>, i8)

define i8 @test_pcmpgt_q_128(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_pcmpgt_q_128
; CHECK: vpcmpgtq %xmm1, %xmm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.q.128(<2 x i64> %a, <2 x i64> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpgt_q_128(<2 x i64> %a, <2 x i64> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_q_128
; CHECK: vpcmpgtq %xmm1, %xmm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.q.128(<2 x i64> %a, <2 x i64> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpgt.q.128(<2 x i64>, <2 x i64>, i8)

define <8 x i8> @test_cmp_d_128(<4 x i32> %a0, <4 x i32> %a1) {
; CHECK-LABEL: test_cmp_d_128
; CHECK: vpcmpeqd %xmm1, %xmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltd %xmm1, %xmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpled %xmm1, %xmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordd %xmm1, %xmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqd %xmm1, %xmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltd %xmm1, %xmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnled %xmm1, %xmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordd %xmm1, %xmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_cmp_d_128(<4 x i32> %a0, <4 x i32> %a1, i8 %mask) {
; CHECK-LABEL: test_mask_cmp_d_128
; CHECK: vpcmpeqd %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltd %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpled %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordd %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqd %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltd %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnled %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordd %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.cmp.d.128(<4 x i32>, <4 x i32>, i32, i8) nounwind readnone

define <8 x i8> @test_ucmp_d_128(<4 x i32> %a0, <4 x i32> %a1) {
; CHECK-LABEL: test_ucmp_d_128
; CHECK: vpcmpequd %xmm1, %xmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltud %xmm1, %xmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleud %xmm1, %xmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordud %xmm1, %xmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequd %xmm1, %xmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltud %xmm1, %xmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleud %xmm1, %xmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordud %xmm1, %xmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_ucmp_d_128(<4 x i32> %a0, <4 x i32> %a1, i8 %mask) {
; CHECK-LABEL: test_mask_ucmp_d_128
; CHECK: vpcmpequd %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltud %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleud %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordud %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequd %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltud %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleud %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordud %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> %a0, <4 x i32> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32>, <4 x i32>, i32, i8) nounwind readnone

define <8 x i8> @test_cmp_q_128(<2 x i64> %a0, <2 x i64> %a1) {
; CHECK-LABEL: test_cmp_q_128
; CHECK: vpcmpeqq %xmm1, %xmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltq %xmm1, %xmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleq %xmm1, %xmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordq %xmm1, %xmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqq %xmm1, %xmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltq %xmm1, %xmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleq %xmm1, %xmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordq %xmm1, %xmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_cmp_q_128(<2 x i64> %a0, <2 x i64> %a1, i8 %mask) {
; CHECK-LABEL: test_mask_cmp_q_128
; CHECK: vpcmpeqq %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltq %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleq %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordq %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqq %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltq %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleq %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordq %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.cmp.q.128(<2 x i64>, <2 x i64>, i32, i8) nounwind readnone

define <8 x i8> @test_ucmp_q_128(<2 x i64> %a0, <2 x i64> %a1) {
; CHECK-LABEL: test_ucmp_q_128
; CHECK: vpcmpequq %xmm1, %xmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuq %xmm1, %xmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuq %xmm1, %xmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduq %xmm1, %xmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequq %xmm1, %xmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuq %xmm1, %xmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuq %xmm1, %xmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduq %xmm1, %xmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_ucmp_q_128(<2 x i64> %a0, <2 x i64> %a1, i8 %mask) {
; CHECK-LABEL: test_mask_ucmp_q_128
; CHECK: vpcmpequq %xmm1, %xmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuq %xmm1, %xmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuq %xmm1, %xmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduq %xmm1, %xmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequq %xmm1, %xmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuq %xmm1, %xmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuq %xmm1, %xmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduq %xmm1, %xmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> %a0, <2 x i64> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64>, <2 x i64>, i32, i8) nounwind readnone

; CHECK-LABEL: compr1
; CHECK: vcompresspd %zmm0
define void @compr1(i8* %addr, <8 x double> %data, i8 %mask) {
  call void @llvm.x86.avx512.mask.compress.store.pd.512(i8* %addr, <8 x double> %data, i8 %mask)
  ret void
}

declare void @llvm.x86.avx512.mask.compress.store.pd.512(i8* %addr, <8 x double> %data, i8 %mask)

; CHECK-LABEL: compr2
; CHECK: vcompresspd %ymm0
define void @compr2(i8* %addr, <4 x double> %data, i8 %mask) {
  call void @llvm.x86.avx512.mask.compress.store.pd.256(i8* %addr, <4 x double> %data, i8 %mask)
  ret void
}

declare void @llvm.x86.avx512.mask.compress.store.pd.256(i8* %addr, <4 x double> %data, i8 %mask)

; CHECK-LABEL: compr3
; CHECK: vcompressps %xmm0
define void @compr3(i8* %addr, <4 x float> %data, i8 %mask) {
  call void @llvm.x86.avx512.mask.compress.store.ps.128(i8* %addr, <4 x float> %data, i8 %mask)
  ret void
}

declare void @llvm.x86.avx512.mask.compress.store.ps.128(i8* %addr, <4 x float> %data, i8 %mask)

; CHECK-LABEL: compr4
; CHECK: vcompresspd %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0xc9,0x8a,0xc0]
define <8 x double> @compr4(i8* %addr, <8 x double> %data, i8 %mask) {
  %res = call <8 x double> @llvm.x86.avx512.mask.compress.pd.512(<8 x double> %data, <8 x double> zeroinitializer, i8 %mask)
  ret <8 x double> %res
}

declare <8 x double> @llvm.x86.avx512.mask.compress.pd.512(<8 x double> %data, <8 x double> %src0, i8 %mask)

; CHECK-LABEL: compr5
; CHECK: vcompresspd %ymm0, %ymm1 {%k1}  ## encoding: [0x62,0xf2,0xfd,0x29,0x8a,0xc1]
define <4 x double> @compr5(<4 x double> %data, <4 x double> %src0, i8 %mask) {
  %res = call <4 x double> @llvm.x86.avx512.mask.compress.pd.256( <4 x double> %data, <4 x double> %src0, i8 %mask)
  ret <4 x double> %res
}

declare <4 x double> @llvm.x86.avx512.mask.compress.pd.256(<4 x double> %data, <4 x double> %src0, i8 %mask)

; CHECK-LABEL: compr6
; CHECK: vcompressps %xmm0
define <4 x float> @compr6(<4 x float> %data, i8 %mask) {
  %res = call <4 x float> @llvm.x86.avx512.mask.compress.ps.128(<4 x float> %data, <4 x float>zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

declare <4 x float> @llvm.x86.avx512.mask.compress.ps.128(<4 x float> %data, <4 x float> %src0, i8 %mask)

; CHECK-LABEL: compr7
; CHECK-NOT: vcompress
; CHECK: vmovapd
define void @compr7(i8* %addr, <8 x double> %data) {
  call void @llvm.x86.avx512.mask.compress.store.pd.512(i8* %addr, <8 x double> %data, i8 -1)
  ret void
}

; CHECK-LABEL: compr8
; CHECK-NOT: vcompressps %xmm0
define <4 x float> @compr8(<4 x float> %data) {
  %res = call <4 x float> @llvm.x86.avx512.mask.compress.ps.128(<4 x float> %data, <4 x float>zeroinitializer, i8 -1)
  ret <4 x float> %res
}

; CHECK-LABEL: compr9
; CHECK: vpcompressq %zmm0, (%rdi) {%k1}  ## encoding: [0x62,0xf2,0xfd,0x49,0x8b,0x07]
define void @compr9(i8* %addr, <8 x i64> %data, i8 %mask) {
  call void @llvm.x86.avx512.mask.compress.store.q.512(i8* %addr, <8 x i64> %data, i8 %mask)
  ret void
}

declare void @llvm.x86.avx512.mask.compress.store.q.512(i8* %addr, <8 x i64> %data, i8 %mask)

; CHECK-LABEL: compr10
; CHECK: vpcompressd %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf2,0x7d,0x89,0x8b,0xc0]
define <4 x i32> @compr10(<4 x i32> %data, i8 %mask) {
  %res = call <4 x i32> @llvm.x86.avx512.mask.compress.d.128(<4 x i32> %data, <4 x i32>zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.compress.d.128(<4 x i32> %data, <4 x i32> %src0, i8 %mask)

; Expand

; CHECK-LABEL: expand1
; CHECK: vexpandpd (%rdi), %zmm0 {%k1}  ## encoding: [0x62,0xf2,0xfd,0x49,0x88,0x07]
define <8 x double> @expand1(i8* %addr, <8 x double> %data, i8 %mask) {
  %res = call <8 x double> @llvm.x86.avx512.mask.expand.load.pd.512(i8* %addr, <8 x double> %data, i8 %mask)
  ret <8 x double> %res
}

declare <8 x double> @llvm.x86.avx512.mask.expand.load.pd.512(i8* %addr, <8 x double> %data, i8 %mask)

; CHECK-LABEL: expand2
; CHECK: vexpandpd (%rdi), %ymm0 {%k1} ## encoding: [0x62,0xf2,0xfd,0x29,0x88,0x07]
define <4 x double> @expand2(i8* %addr, <4 x double> %data, i8 %mask) {
  %res = call <4 x double> @llvm.x86.avx512.mask.expand.load.pd.256(i8* %addr, <4 x double> %data, i8 %mask)
  ret <4 x double> %res
}

declare <4 x double> @llvm.x86.avx512.mask.expand.load.pd.256(i8* %addr, <4 x double> %data, i8 %mask)

; CHECK-LABEL: expand3
; CHECK: vexpandps (%rdi), %xmm0 {%k1} ## encoding: [0x62,0xf2,0x7d,0x09,0x88,0x07]
define <4 x float> @expand3(i8* %addr, <4 x float> %data, i8 %mask) {
  %res = call <4 x float> @llvm.x86.avx512.mask.expand.load.ps.128(i8* %addr, <4 x float> %data, i8 %mask)
  ret <4 x float> %res
}

declare <4 x float> @llvm.x86.avx512.mask.expand.load.ps.128(i8* %addr, <4 x float> %data, i8 %mask)

; CHECK-LABEL: expand4
; CHECK: vexpandpd %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0xc9,0x88,0xc0]
define <8 x double> @expand4(i8* %addr, <8 x double> %data, i8 %mask) {
  %res = call <8 x double> @llvm.x86.avx512.mask.expand.pd.512(<8 x double> %data, <8 x double> zeroinitializer, i8 %mask)
  ret <8 x double> %res
}

declare <8 x double> @llvm.x86.avx512.mask.expand.pd.512(<8 x double> %data, <8 x double> %src0, i8 %mask)

; CHECK-LABEL: expand5
; CHECK: vexpandpd %ymm0, %ymm1 {%k1}  ## encoding: [0x62,0xf2,0xfd,0x29,0x88,0xc8]
define <4 x double> @expand5(<4 x double> %data, <4 x double> %src0, i8 %mask) {
  %res = call <4 x double> @llvm.x86.avx512.mask.expand.pd.256( <4 x double> %data, <4 x double> %src0, i8 %mask)
  ret <4 x double> %res
}

declare <4 x double> @llvm.x86.avx512.mask.expand.pd.256(<4 x double> %data, <4 x double> %src0, i8 %mask)

; CHECK-LABEL: expand6
; CHECK: vexpandps %xmm0
define <4 x float> @expand6(<4 x float> %data, i8 %mask) {
  %res = call <4 x float> @llvm.x86.avx512.mask.expand.ps.128(<4 x float> %data, <4 x float>zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

declare <4 x float> @llvm.x86.avx512.mask.expand.ps.128(<4 x float> %data, <4 x float> %src0, i8 %mask)

; CHECK-LABEL: expand7
; CHECK-NOT: vexpand
; CHECK: vmovapd
define <8 x double> @expand7(i8* %addr, <8 x double> %data) {
  %res = call <8 x double> @llvm.x86.avx512.mask.expand.load.pd.512(i8* %addr, <8 x double> %data, i8 -1)
  ret <8 x double> %res
}

; CHECK-LABEL: expand8
; CHECK-NOT: vexpandps %xmm0
define <4 x float> @expand8(<4 x float> %data) {
  %res = call <4 x float> @llvm.x86.avx512.mask.expand.ps.128(<4 x float> %data, <4 x float>zeroinitializer, i8 -1)
  ret <4 x float> %res
}

; CHECK-LABEL: expand9
; CHECK: vpexpandq (%rdi), %zmm0 {%k1} ## encoding: [0x62,0xf2,0xfd,0x49,0x89,0x07]
define <8 x i64> @expand9(i8* %addr, <8 x i64> %data, i8 %mask) {
  %res = call <8 x i64> @llvm.x86.avx512.mask.expand.load.q.512(i8* %addr, <8 x i64> %data, i8 %mask)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.expand.load.q.512(i8* %addr, <8 x i64> %data, i8 %mask)

; CHECK-LABEL: expand10
; CHECK: vpexpandd %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf2,0x7d,0x89,0x89,0xc0]
define <4 x i32> @expand10(<4 x i32> %data, i8 %mask) {
  %res = call <4 x i32> @llvm.x86.avx512.mask.expand.d.128(<4 x i32> %data, <4 x i32>zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.expand.d.128(<4 x i32> %data, <4 x i32> %src0, i8 %mask)

define <8 x float> @test_x86_mask_blend_ps_256(i8 %a0, <8 x float> %a1, <8 x float> %a2) {
  ; CHECK: vblendmps %ymm1, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.mask.blend.ps.256(<8 x float> %a1, <8 x float> %a2, i8 %a0) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}

declare <8 x float> @llvm.x86.avx512.mask.blend.ps.256(<8 x float>, <8 x float>, i8) nounwind readonly

define <4 x double> @test_x86_mask_blend_pd_256(i8 %a0, <4 x double> %a1, <4 x double> %a2) {
  ; CHECK: vblendmpd %ymm1, %ymm0
  %res = call <4 x double> @llvm.x86.avx512.mask.blend.pd.256(<4 x double> %a1, <4 x double> %a2, i8 %a0) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}

define <4 x double> @test_x86_mask_blend_pd_256_memop(<4 x double> %a, <4 x double>* %ptr, i8 %mask) {
  ; CHECK-LABEL: test_x86_mask_blend_pd_256_memop
  ; CHECK: vblendmpd (%
  %b = load <4 x double>, <4 x double>* %ptr
  %res = call <4 x double> @llvm.x86.avx512.mask.blend.pd.256(<4 x double> %a, <4 x double> %b, i8 %mask) ; <<4 x double>> [#uses=1]
  ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx512.mask.blend.pd.256(<4 x double>, <4 x double>, i8) nounwind readonly

; CHECK-LABEL: test_x86_mask_blend_d_256
; CHECK: vpblendmd
define <8 x i32> @test_x86_mask_blend_d_256(i8 %a0, <8 x i32> %a1, <8 x i32> %a2) {
  %res = call <8 x i32> @llvm.x86.avx512.mask.blend.d.256(<8 x i32> %a1, <8 x i32> %a2, i8 %a0) ; <<8 x i32>> [#uses=1]
  ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx512.mask.blend.d.256(<8 x i32>, <8 x i32>, i8) nounwind readonly

define <4 x i64> @test_x86_mask_blend_q_256(i8 %a0, <4 x i64> %a1, <4 x i64> %a2) {
  ; CHECK: vpblendmq
  %res = call <4 x i64> @llvm.x86.avx512.mask.blend.q.256(<4 x i64> %a1, <4 x i64> %a2, i8 %a0) ; <<4 x i64>> [#uses=1]
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.avx512.mask.blend.q.256(<4 x i64>, <4 x i64>, i8) nounwind readonly

define <4 x float> @test_x86_mask_blend_ps_128(i8 %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: vblendmps %xmm1, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.blend.ps.128(<4 x float> %a1, <4 x float> %a2, i8 %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}

declare <4 x float> @llvm.x86.avx512.mask.blend.ps.128(<4 x float>, <4 x float>, i8) nounwind readonly

define <2 x double> @test_x86_mask_blend_pd_128(i8 %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: vblendmpd %xmm1, %xmm0
  %res = call <2 x double> @llvm.x86.avx512.mask.blend.pd.128(<2 x double> %a1, <2 x double> %a2, i8 %a0) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}

define <2 x double> @test_x86_mask_blend_pd_128_memop(<2 x double> %a, <2 x double>* %ptr, i8 %mask) {
  ; CHECK-LABEL: test_x86_mask_blend_pd_128_memop
  ; CHECK: vblendmpd (%
  %b = load <2 x double>, <2 x double>* %ptr
  %res = call <2 x double> @llvm.x86.avx512.mask.blend.pd.128(<2 x double> %a, <2 x double> %b, i8 %mask) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.avx512.mask.blend.pd.128(<2 x double>, <2 x double>, i8) nounwind readonly

define <4 x i32> @test_x86_mask_blend_d_128(i8 %a0, <4 x i32> %a1, <4 x i32> %a2) {
  ; CHECK: vpblendmd
  %res = call <4 x i32> @llvm.x86.avx512.mask.blend.d.128(<4 x i32> %a1, <4 x i32> %a2, i8 %a0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.x86.avx512.mask.blend.d.128(<4 x i32>, <4 x i32>, i8) nounwind readonly

define <2 x i64> @test_x86_mask_blend_q_128(i8 %a0, <2 x i64> %a1, <2 x i64> %a2) {
  ; CHECK: vpblendmq
  %res = call <2 x i64> @llvm.x86.avx512.mask.blend.q.128(<2 x i64> %a1, <2 x i64> %a2, i8 %a0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.avx512.mask.blend.q.128(<2 x i64>, <2 x i64>, i8) nounwind readonly


define < 2 x i64> @test_mask_mul_epi32_rr_128(< 4 x i32> %a, < 4 x i32> %b) {
  ;CHECK-LABEL: test_mask_mul_epi32_rr_128
  ;CHECK: vpmuldq %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf2,0xfd,0x08,0x28,0xc1]
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 -1)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epi32_rrk_128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rrk_128
  ;CHECK: vpmuldq %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf2,0xfd,0x09,0x28,0xd1]
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> %passThru, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epi32_rrkz_128(< 4 x i32> %a, < 4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rrkz_128
  ;CHECK: vpmuldq %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0x89,0x28,0xc1]
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epi32_rm_128(< 4 x i32> %a, < 4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epi32_rm_128
  ;CHECK: vpmuldq (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf2,0xfd,0x08,0x28,0x07]
  %b = load < 4 x i32>, < 4 x i32>* %ptr_b
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 -1)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epi32_rmk_128(< 4 x i32> %a, < 4 x i32>* %ptr_b, < 2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmk_128
  ;CHECK: vpmuldq (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf2,0xfd,0x09,0x28,0x0f]
  %b = load < 4 x i32>, < 4 x i32>* %ptr_b
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> %passThru, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epi32_rmkz_128(< 4 x i32> %a, < 4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmkz_128
  ;CHECK: vpmuldq (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0x89,0x28,0x07]
  %b = load < 4 x i32>, < 4 x i32>* %ptr_b
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epi32_rmb_128(< 4 x i32> %a, i64* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmb_128
  ;CHECK: vpmuldq (%rdi){1to2}, %xmm0, %xmm0  ## encoding: [0x62,0xf2,0xfd,0x18,0x28,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 2 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 2 x i64> %vecinit.i, < 2 x i64> undef, <2 x i32> zeroinitializer
  %b = bitcast < 2 x i64> %b64 to < 4 x i32>
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 -1)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epi32_rmbk_128(< 4 x i32> %a, i64* %ptr_b, < 2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmbk_128
  ;CHECK: vpmuldq (%rdi){1to2}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf2,0xfd,0x19,0x28,0x0f]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 2 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 2 x i64> %vecinit.i, < 2 x i64> undef, <2 x i32> zeroinitializer
  %b = bitcast < 2 x i64> %b64 to < 4 x i32>
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> %passThru, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epi32_rmbkz_128(< 4 x i32> %a, i64* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmbkz_128
  ;CHECK: vpmuldq (%rdi){1to2}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0x99,0x28,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 2 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 2 x i64> %vecinit.i, < 2 x i64> undef, < 2 x i32> zeroinitializer
  %b = bitcast < 2 x i64> %b64 to < 4 x i32>
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 %mask)
  ret < 2 x i64> %res
}

declare < 2 x i64> @llvm.x86.avx512.mask.pmul.dq.128(< 4 x i32>, < 4 x i32>, < 2 x i64>, i8)

define < 4 x i64> @test_mask_mul_epi32_rr_256(< 8 x i32> %a, < 8 x i32> %b) {
  ;CHECK-LABEL: test_mask_mul_epi32_rr_256
  ;CHECK: vpmuldq %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf2,0xfd,0x28,0x28,0xc1]
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 -1)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epi32_rrk_256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rrk_256
  ;CHECK: vpmuldq %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf2,0xfd,0x29,0x28,0xd1]
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> %passThru, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epi32_rrkz_256(< 8 x i32> %a, < 8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rrkz_256
  ;CHECK: vpmuldq %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0xa9,0x28,0xc1]
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epi32_rm_256(< 8 x i32> %a, < 8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epi32_rm_256
  ;CHECK: vpmuldq (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf2,0xfd,0x28,0x28,0x07]
  %b = load < 8 x i32>, < 8 x i32>* %ptr_b
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 -1)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epi32_rmk_256(< 8 x i32> %a, < 8 x i32>* %ptr_b, < 4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmk_256
  ;CHECK: vpmuldq (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf2,0xfd,0x29,0x28,0x0f]
  %b = load < 8 x i32>, < 8 x i32>* %ptr_b
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> %passThru, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epi32_rmkz_256(< 8 x i32> %a, < 8 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmkz_256
  ;CHECK: vpmuldq (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0xa9,0x28,0x07]
  %b = load < 8 x i32>, < 8 x i32>* %ptr_b
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epi32_rmb_256(< 8 x i32> %a, i64* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmb_256
  ;CHECK: vpmuldq (%rdi){1to4}, %ymm0, %ymm0  ## encoding: [0x62,0xf2,0xfd,0x38,0x28,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 4 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 4 x i64> %vecinit.i, < 4 x i64> undef, < 4 x i32> zeroinitializer
  %b = bitcast < 4 x i64> %b64 to < 8 x i32>
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 -1)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epi32_rmbk_256(< 8 x i32> %a, i64* %ptr_b, < 4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmbk_256
  ;CHECK: vpmuldq (%rdi){1to4}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf2,0xfd,0x39,0x28,0x0f]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 4 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 4 x i64> %vecinit.i, < 4 x i64> undef, < 4 x i32> zeroinitializer
  %b = bitcast < 4 x i64> %b64 to < 8 x i32>
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> %passThru, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epi32_rmbkz_256(< 8 x i32> %a, i64* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epi32_rmbkz_256
  ;CHECK: vpmuldq (%rdi){1to4}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf2,0xfd,0xb9,0x28,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 4 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 4 x i64> %vecinit.i, < 4 x i64> undef, < 4 x i32> zeroinitializer
  %b = bitcast < 4 x i64> %b64 to < 8 x i32>
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 %mask)
  ret < 4 x i64> %res
}

declare < 4 x i64> @llvm.x86.avx512.mask.pmul.dq.256(< 8 x i32>, < 8 x i32>, < 4 x i64>, i8)

define < 2 x i64> @test_mask_mul_epu32_rr_128(< 4 x i32> %a, < 4 x i32> %b) {
  ;CHECK-LABEL: test_mask_mul_epu32_rr_128
  ;CHECK: vpmuludq %xmm1, %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0xf4,0xc1]
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 -1)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epu32_rrk_128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rrk_128
  ;CHECK: vpmuludq %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x09,0xf4,0xd1]
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> %passThru, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epu32_rrkz_128(< 4 x i32> %a, < 4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rrkz_128
  ;CHECK: vpmuludq %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x89,0xf4,0xc1]
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epu32_rm_128(< 4 x i32> %a, < 4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epu32_rm_128
  ;CHECK: vpmuludq (%rdi), %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0xf4,0x07]
  %b = load < 4 x i32>, < 4 x i32>* %ptr_b
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 -1)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epu32_rmk_128(< 4 x i32> %a, < 4 x i32>* %ptr_b, < 2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmk_128
  ;CHECK: vpmuludq (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x09,0xf4,0x0f]
  %b = load < 4 x i32>, < 4 x i32>* %ptr_b
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> %passThru, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epu32_rmkz_128(< 4 x i32> %a, < 4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmkz_128
  ;CHECK: vpmuludq (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x89,0xf4,0x07]
  %b = load < 4 x i32>, < 4 x i32>* %ptr_b
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epu32_rmb_128(< 4 x i32> %a, i64* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmb_128
  ;CHECK: vpmuludq (%rdi){1to2}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0xfd,0x18,0xf4,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 2 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 2 x i64> %vecinit.i, < 2 x i64> undef, <2 x i32> zeroinitializer
  %b = bitcast < 2 x i64> %b64 to < 4 x i32>
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 -1)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epu32_rmbk_128(< 4 x i32> %a, i64* %ptr_b, < 2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmbk_128
  ;CHECK: vpmuludq (%rdi){1to2}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x19,0xf4,0x0f]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 2 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 2 x i64> %vecinit.i, < 2 x i64> undef, <2 x i32> zeroinitializer
  %b = bitcast < 2 x i64> %b64 to < 4 x i32>
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> %passThru, i8 %mask)
  ret < 2 x i64> %res
}

define < 2 x i64> @test_mask_mul_epu32_rmbkz_128(< 4 x i32> %a, i64* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmbkz_128
  ;CHECK: vpmuludq (%rdi){1to2}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x99,0xf4,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 2 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 2 x i64> %vecinit.i, < 2 x i64> undef, < 2 x i32> zeroinitializer
  %b = bitcast < 2 x i64> %b64 to < 4 x i32>
  %res = call < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32> %a, < 4 x i32> %b, < 2 x i64> zeroinitializer, i8 %mask)
  ret < 2 x i64> %res
}

declare < 2 x i64> @llvm.x86.avx512.mask.pmulu.dq.128(< 4 x i32>, < 4 x i32>, < 2 x i64>, i8)

define < 4 x i64> @test_mask_mul_epu32_rr_256(< 8 x i32> %a, < 8 x i32> %b) {
  ;CHECK-LABEL: test_mask_mul_epu32_rr_256
  ;CHECK: vpmuludq %ymm1, %ymm0, %ymm0 ## encoding: [0xc5,0xfd,0xf4,0xc1]
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 -1)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epu32_rrk_256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rrk_256
  ;CHECK: vpmuludq %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x29,0xf4,0xd1]
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> %passThru, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epu32_rrkz_256(< 8 x i32> %a, < 8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rrkz_256
  ;CHECK: vpmuludq %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xa9,0xf4,0xc1]
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epu32_rm_256(< 8 x i32> %a, < 8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epu32_rm_256
  ;CHECK: vpmuludq (%rdi), %ymm0, %ymm0 ## encoding: [0xc5,0xfd,0xf4,0x07]
  %b = load < 8 x i32>, < 8 x i32>* %ptr_b
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 -1)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epu32_rmk_256(< 8 x i32> %a, < 8 x i32>* %ptr_b, < 4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmk_256
  ;CHECK: vpmuludq (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x29,0xf4,0x0f]
  %b = load < 8 x i32>, < 8 x i32>* %ptr_b
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> %passThru, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epu32_rmkz_256(< 8 x i32> %a, < 8 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmkz_256
  ;CHECK: vpmuludq (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xa9,0xf4,0x07]
  %b = load < 8 x i32>, < 8 x i32>* %ptr_b
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epu32_rmb_256(< 8 x i32> %a, i64* %ptr_b) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmb_256
  ;CHECK: vpmuludq (%rdi){1to4}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0xfd,0x38,0xf4,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 4 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 4 x i64> %vecinit.i, < 4 x i64> undef, < 4 x i32> zeroinitializer
  %b = bitcast < 4 x i64> %b64 to < 8 x i32>
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 -1)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epu32_rmbk_256(< 8 x i32> %a, i64* %ptr_b, < 4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmbk_256
  ;CHECK: vpmuludq (%rdi){1to4}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x39,0xf4,0x0f]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 4 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 4 x i64> %vecinit.i, < 4 x i64> undef, < 4 x i32> zeroinitializer
  %b = bitcast < 4 x i64> %b64 to < 8 x i32>
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> %passThru, i8 %mask)
  ret < 4 x i64> %res
}

define < 4 x i64> @test_mask_mul_epu32_rmbkz_256(< 8 x i32> %a, i64* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_mul_epu32_rmbkz_256
  ;CHECK: vpmuludq (%rdi){1to4}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xb9,0xf4,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement < 4 x i64> undef, i64 %q, i32 0
  %b64 = shufflevector < 4 x i64> %vecinit.i, < 4 x i64> undef, < 4 x i32> zeroinitializer
  %b = bitcast < 4 x i64> %b64 to < 8 x i32>
  %res = call < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32> %a, < 8 x i32> %b, < 4 x i64> zeroinitializer, i8 %mask)
  ret < 4 x i64> %res
}

declare < 4 x i64> @llvm.x86.avx512.mask.pmulu.dq.256(< 8 x i32>, < 8 x i32>, < 4 x i64>, i8)

define <4 x i32> @test_mask_add_epi32_rr_128(<4 x i32> %a, <4 x i32> %b) {
  ;CHECK-LABEL: test_mask_add_epi32_rr_128
  ;CHECK: vpaddd %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xfe,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_add_epi32_rrk_128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rrk_128
  ;CHECK: vpaddd %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xfe,0xd1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_add_epi32_rrkz_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rrkz_128
  ;CHECK: vpaddd %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xfe,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_add_epi32_rm_128(<4 x i32> %a, <4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi32_rm_128
  ;CHECK: vpaddd (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xfe,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_add_epi32_rmk_128(<4 x i32> %a, <4 x i32>* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmk_128
  ;CHECK: vpaddd (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xfe,0x0f]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_add_epi32_rmkz_128(<4 x i32> %a, <4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmkz_128
  ;CHECK: vpaddd (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xfe,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_add_epi32_rmb_128(<4 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi32_rmb_128
  ;CHECK: vpaddd (%rdi){1to4}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0x7d,0x18,0xfe,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_add_epi32_rmbk_128(<4 x i32> %a, i32* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmbk_128
  ;CHECK: vpaddd (%rdi){1to4}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x19,0xfe,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_add_epi32_rmbkz_128(<4 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmbkz_128
  ;CHECK: vpaddd (%rdi){1to4}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x99,0xfe,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.padd.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32> @test_mask_sub_epi32_rr_128(<4 x i32> %a, <4 x i32> %b) {
  ;CHECK-LABEL: test_mask_sub_epi32_rr_128
  ;CHECK: vpsubd %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xfa,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_sub_epi32_rrk_128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rrk_128
  ;CHECK: vpsubd %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xfa,0xd1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_sub_epi32_rrkz_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rrkz_128
  ;CHECK: vpsubd %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xfa,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_sub_epi32_rm_128(<4 x i32> %a, <4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi32_rm_128
  ;CHECK: (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xfa,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_sub_epi32_rmk_128(<4 x i32> %a, <4 x i32>* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmk_128
  ;CHECK: vpsubd (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xfa,0x0f]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_sub_epi32_rmkz_128(<4 x i32> %a, <4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmkz_128
  ;CHECK: vpsubd (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xfa,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_sub_epi32_rmb_128(<4 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmb_128
  ;CHECK: vpsubd (%rdi){1to4}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0x7d,0x18,0xfa,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_sub_epi32_rmbk_128(<4 x i32> %a, i32* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmbk_128
  ;CHECK: vpsubd (%rdi){1to4}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x19,0xfa,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_sub_epi32_rmbkz_128(<4 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmbkz_128
  ;CHECK: vpsubd (%rdi){1to4}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x99,0xfa,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.psub.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <8 x i32> @test_mask_sub_epi32_rr_256(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: test_mask_sub_epi32_rr_256
  ;CHECK: vpsubd %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xfa,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_sub_epi32_rrk_256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rrk_256
  ;CHECK: vpsubd %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xfa,0xd1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_sub_epi32_rrkz_256(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rrkz_256
  ;CHECK: vpsubd %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xfa,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_sub_epi32_rm_256(<8 x i32> %a, <8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi32_rm_256
  ;CHECK: vpsubd (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xfa,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_sub_epi32_rmk_256(<8 x i32> %a, <8 x i32>* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmk_256
  ;CHECK: vpsubd (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xfa,0x0f]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_sub_epi32_rmkz_256(<8 x i32> %a, <8 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmkz_256
  ;CHECK: vpsubd (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xfa,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_sub_epi32_rmb_256(<8 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmb_256
  ;CHECK: vpsubd (%rdi){1to8}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0x7d,0x38,0xfa,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_sub_epi32_rmbk_256(<8 x i32> %a, i32* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmbk_256
  ;CHECK: vpsubd (%rdi){1to8}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x39,0xfa,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_sub_epi32_rmbkz_256(<8 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_sub_epi32_rmbkz_256
  ;CHECK: vpsubd (%rdi){1to8}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xb9,0xfa,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

declare <8 x i32> @llvm.x86.avx512.mask.psub.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32> @test_mask_add_epi32_rr_256(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: test_mask_add_epi32_rr_256
  ;CHECK: vpaddd %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xfe,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_add_epi32_rrk_256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rrk_256
  ;CHECK: vpaddd %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xfe,0xd1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_add_epi32_rrkz_256(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rrkz_256
  ;CHECK: vpaddd %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xfe,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_add_epi32_rm_256(<8 x i32> %a, <8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi32_rm_256
  ;CHECK: vpaddd (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xfe,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_add_epi32_rmk_256(<8 x i32> %a, <8 x i32>* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmk_256
  ;CHECK: vpaddd (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xfe,0x0f]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_add_epi32_rmkz_256(<8 x i32> %a, <8 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmkz_256
  ;CHECK: vpaddd (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xfe,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_add_epi32_rmb_256(<8 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_add_epi32_rmb_256
  ;CHECK: vpaddd (%rdi){1to8}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0x7d,0x38,0xfe,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_add_epi32_rmbk_256(<8 x i32> %a, i32* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmbk_256
  ;CHECK: vpaddd (%rdi){1to8}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x39,0xfe,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_add_epi32_rmbkz_256(<8 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_add_epi32_rmbkz_256
  ;CHECK: vpaddd (%rdi){1to8}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xb9,0xfe,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

declare <8 x i32> @llvm.x86.avx512.mask.padd.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <4 x i32> @test_mask_and_epi32_rr_128(<4 x i32> %a, <4 x i32> %b) {
  ;CHECK-LABEL: test_mask_and_epi32_rr_128
  ;CHECK: vpandd  %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xdb,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_and_epi32_rrk_128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rrk_128
  ;CHECK: vpandd  %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xdb,0xd1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_and_epi32_rrkz_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rrkz_128
  ;CHECK: vpandd  %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xdb,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_and_epi32_rm_128(<4 x i32> %a, <4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_and_epi32_rm_128
  ;CHECK: vpandd  (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xdb,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_and_epi32_rmk_128(<4 x i32> %a, <4 x i32>* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rmk_128
  ;CHECK: vpandd  (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xdb,0x0f]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_and_epi32_rmkz_128(<4 x i32> %a, <4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rmkz_128
  ;CHECK: vpandd  (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xdb,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_and_epi32_rmb_128(<4 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_and_epi32_rmb_128
  ;CHECK: vpandd  (%rdi){1to4}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0x7d,0x18,0xdb,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_and_epi32_rmbk_128(<4 x i32> %a, i32* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rmbk_128
  ;CHECK: vpandd  (%rdi){1to4}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x19,0xdb,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_and_epi32_rmbkz_128(<4 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rmbkz_128
  ;CHECK: vpandd  (%rdi){1to4}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x99,0xdb,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.pand.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <8 x i32> @test_mask_and_epi32_rr_256(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: test_mask_and_epi32_rr_256
  ;CHECK: vpandd  %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xdb,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_and_epi32_rrk_256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rrk_256
  ;CHECK: vpandd  %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xdb,0xd1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_and_epi32_rrkz_256(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rrkz_256
  ;CHECK: vpandd  %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xdb,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_and_epi32_rm_256(<8 x i32> %a, <8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_and_epi32_rm_256
  ;CHECK: vpandd  (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xdb,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_and_epi32_rmk_256(<8 x i32> %a, <8 x i32>* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rmk_256
  ;CHECK: vpandd  (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xdb,0x0f]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_and_epi32_rmkz_256(<8 x i32> %a, <8 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rmkz_256
  ;CHECK: vpandd  (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xdb,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_and_epi32_rmb_256(<8 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_and_epi32_rmb_256
  ;CHECK: vpandd  (%rdi){1to8}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0x7d,0x38,0xdb,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_and_epi32_rmbk_256(<8 x i32> %a, i32* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rmbk_256
  ;CHECK: vpandd  (%rdi){1to8}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x39,0xdb,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_and_epi32_rmbkz_256(<8 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_and_epi32_rmbkz_256
  ;CHECK: vpandd  (%rdi){1to8}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xb9,0xdb,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

declare <8 x i32> @llvm.x86.avx512.mask.pand.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <4 x i32> @test_mask_or_epi32_rr_128(<4 x i32> %a, <4 x i32> %b) {
  ;CHECK-LABEL: test_mask_or_epi32_rr_128
  ;CHECK: vpord   %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xeb,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_or_epi32_rrk_128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rrk_128
  ;CHECK: vpord   %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xeb,0xd1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_or_epi32_rrkz_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rrkz_128
  ;CHECK: vpord   %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xeb,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_or_epi32_rm_128(<4 x i32> %a, <4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_or_epi32_rm_128
  ;CHECK: vpord   (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xeb,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_or_epi32_rmk_128(<4 x i32> %a, <4 x i32>* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rmk_128
  ;CHECK: vpord   (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xeb,0x0f]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_or_epi32_rmkz_128(<4 x i32> %a, <4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rmkz_128
  ;CHECK: vpord   (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xeb,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_or_epi32_rmb_128(<4 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_or_epi32_rmb_128
  ;CHECK: vpord   (%rdi){1to4}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0x7d,0x18,0xeb,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_or_epi32_rmbk_128(<4 x i32> %a, i32* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rmbk_128
  ;CHECK: vpord   (%rdi){1to4}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x19,0xeb,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_or_epi32_rmbkz_128(<4 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rmbkz_128
  ;CHECK: vpord   (%rdi){1to4}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x99,0xeb,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.por.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <8 x i32> @test_mask_or_epi32_rr_256(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: test_mask_or_epi32_rr_256
  ;CHECK: vpord   %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xeb,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_or_epi32_rrk_256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rrk_256
  ;CHECK: vpord   %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xeb,0xd1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_or_epi32_rrkz_256(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rrkz_256
  ;CHECK: vpord   %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xeb,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_or_epi32_rm_256(<8 x i32> %a, <8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_or_epi32_rm_256
  ;CHECK: vpord   (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xeb,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_or_epi32_rmk_256(<8 x i32> %a, <8 x i32>* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rmk_256
  ;CHECK: vpord   (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xeb,0x0f]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_or_epi32_rmkz_256(<8 x i32> %a, <8 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rmkz_256
  ;CHECK: vpord   (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xeb,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_or_epi32_rmb_256(<8 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_or_epi32_rmb_256
  ;CHECK: vpord   (%rdi){1to8}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0x7d,0x38,0xeb,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_or_epi32_rmbk_256(<8 x i32> %a, i32* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rmbk_256
  ;CHECK: vpord   (%rdi){1to8}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x39,0xeb,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_or_epi32_rmbkz_256(<8 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_or_epi32_rmbkz_256
  ;CHECK: vpord   (%rdi){1to8}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xb9,0xeb,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

declare <8 x i32> @llvm.x86.avx512.mask.por.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <4 x i32> @test_mask_xor_epi32_rr_128(<4 x i32> %a, <4 x i32> %b) {
  ;CHECK-LABEL: test_mask_xor_epi32_rr_128
  ;CHECK: vpxord  %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xef,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_xor_epi32_rrk_128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rrk_128
  ;CHECK: vpxord  %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xef,0xd1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_xor_epi32_rrkz_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rrkz_128
  ;CHECK: vpxord  %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xef,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_xor_epi32_rm_128(<4 x i32> %a, <4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_xor_epi32_rm_128
  ;CHECK: vpxord  (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xef,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_xor_epi32_rmk_128(<4 x i32> %a, <4 x i32>* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmk_128
  ;CHECK: vpxord  (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xef,0x0f]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_xor_epi32_rmkz_128(<4 x i32> %a, <4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmkz_128
  ;CHECK: vpxord  (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xef,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_xor_epi32_rmb_128(<4 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmb_128
  ;CHECK: vpxord  (%rdi){1to4}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0x7d,0x18,0xef,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_xor_epi32_rmbk_128(<4 x i32> %a, i32* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmbk_128
  ;CHECK: vpxord  (%rdi){1to4}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x19,0xef,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_xor_epi32_rmbkz_128(<4 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmbkz_128
  ;CHECK: vpxord  (%rdi){1to4}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x99,0xef,0x07]  
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.pxor.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <8 x i32> @test_mask_xor_epi32_rr_256(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: test_mask_xor_epi32_rr_256
  ;CHECK: vpxord  %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xef,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_xor_epi32_rrk_256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rrk_256
  ;CHECK: vpxord  %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xef,0xd1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_xor_epi32_rrkz_256(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rrkz_256
  ;CHECK: vpxord  %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xef,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_xor_epi32_rm_256(<8 x i32> %a, <8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_xor_epi32_rm_256
  ;CHECK: vpxord  (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xef,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_xor_epi32_rmk_256(<8 x i32> %a, <8 x i32>* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmk_256
  ;CHECK: vpxord  (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xef,0x0f]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_xor_epi32_rmkz_256(<8 x i32> %a, <8 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmkz_256
  ;CHECK: vpxord  (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xef,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_xor_epi32_rmb_256(<8 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmb_256
  ;CHECK: vpxord  (%rdi){1to8}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0x7d,0x38,0xef,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_xor_epi32_rmbk_256(<8 x i32> %a, i32* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmbk_256
  ;CHECK: vpxord  (%rdi){1to8}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x39,0xef,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_xor_epi32_rmbkz_256(<8 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_xor_epi32_rmbkz_256
  ;CHECK: vpxord  (%rdi){1to8}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xb9,0xef,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

declare <8 x i32> @llvm.x86.avx512.mask.pxor.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <4 x i32> @test_mask_andnot_epi32_rr_128(<4 x i32> %a, <4 x i32> %b) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rr_128
  ;CHECK: vpandnd  %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0x7d,0x08,0xdf,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_andnot_epi32_rrk_128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rrk_128
  ;CHECK: vpandnd  %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xdf,0xd1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_andnot_epi32_rrkz_128(<4 x i32> %a, <4 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rrkz_128
  ;CHECK: vpandnd  %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xdf,0xc1]
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_andnot_epi32_rm_128(<4 x i32> %a, <4 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rm_128
  ;CHECK: vpandnd  (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0x7d,0x08,0xdf,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_andnot_epi32_rmk_128(<4 x i32> %a, <4 x i32>* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmk_128
  ;CHECK: vpandnd  (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x09,0xdf,0x0f]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_andnot_epi32_rmkz_128(<4 x i32> %a, <4 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmkz_128
  ;CHECK: vpandnd  (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x89,0xdf,0x07]
  %b = load <4 x i32>, <4 x i32>* %ptr_b
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_andnot_epi32_rmb_128(<4 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmb_128
  ;CHECK: vpandnd  (%rdi){1to4}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0x7d,0x18,0xdf,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 -1)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_andnot_epi32_rmbk_128(<4 x i32> %a, i32* %ptr_b, <4 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmbk_128
  ;CHECK: vpandnd  (%rdi){1to4}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x19,0xdf,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> %passThru, i8 %mask)
  ret <4 x i32> %res
}

define <4 x i32> @test_mask_andnot_epi32_rmbkz_128(<4 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmbkz_128
  ;CHECK: vpandnd  (%rdi){1to4}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0x99,0xdf,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %b = shufflevector <4 x i32> %vecinit.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %res = call <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32> %a, <4 x i32> %b, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.pandn.d.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <8 x i32> @test_mask_andnot_epi32_rr_256(<8 x i32> %a, <8 x i32> %b) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rr_256
  ;CHECK: vpandnd  %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0x7d,0x28,0xdf,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_andnot_epi32_rrk_256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rrk_256
  ;CHECK: vpandnd  %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xdf,0xd1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_andnot_epi32_rrkz_256(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rrkz_256
  ;CHECK: vpandnd  %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xdf,0xc1]
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_andnot_epi32_rm_256(<8 x i32> %a, <8 x i32>* %ptr_b) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rm_256
  ;CHECK: vpandnd  (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0x7d,0x28,0xdf,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_andnot_epi32_rmk_256(<8 x i32> %a, <8 x i32>* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmk_256
  ;CHECK: vpandnd  (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x29,0xdf,0x0f]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_andnot_epi32_rmkz_256(<8 x i32> %a, <8 x i32>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmkz_256
  ;CHECK: vpandnd  (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xa9,0xdf,0x07]
  %b = load <8 x i32>, <8 x i32>* %ptr_b
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_andnot_epi32_rmb_256(<8 x i32> %a, i32* %ptr_b) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmb_256
  ;CHECK: vpandnd  (%rdi){1to8}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0x7d,0x38,0xdf,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 -1)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_andnot_epi32_rmbk_256(<8 x i32> %a, i32* %ptr_b, <8 x i32> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmbk_256
  ;CHECK: vpandnd  (%rdi){1to8}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0x7d,0x39,0xdf,0x0f]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> %passThru, i8 %mask)
  ret <8 x i32> %res
}

define <8 x i32> @test_mask_andnot_epi32_rmbkz_256(<8 x i32> %a, i32* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi32_rmbkz_256
  ;CHECK: vpandnd  (%rdi){1to8}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0x7d,0xb9,0xdf,0x07]
  %q = load i32, i32* %ptr_b
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %b = shufflevector <8 x i32> %vecinit.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %res = call <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32> %a, <8 x i32> %b, <8 x i32> zeroinitializer, i8 %mask)
  ret <8 x i32> %res
}

declare <8 x i32> @llvm.x86.avx512.mask.pandn.d.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <2 x i64> @test_mask_andnot_epi64_rr_128(<2 x i64> %a, <2 x i64> %b) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rr_128
  ;CHECK: vpandnq  %xmm1, %xmm0, %xmm0     ## encoding: [0x62,0xf1,0xfd,0x08,0xdf,0xc1]
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> zeroinitializer, i8 -1)
  ret <2 x i64> %res
}

define <2 x i64> @test_mask_andnot_epi64_rrk_128(<2 x i64> %a, <2 x i64> %b, <2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rrk_128
  ;CHECK: vpandnq  %xmm1, %xmm0, %xmm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x09,0xdf,0xd1]
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> %passThru, i8 %mask)
  ret <2 x i64> %res
}

define <2 x i64> @test_mask_andnot_epi64_rrkz_128(<2 x i64> %a, <2 x i64> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rrkz_128
  ;CHECK: vpandnq  %xmm1, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x89,0xdf,0xc1]
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> zeroinitializer, i8 %mask)
  ret <2 x i64> %res
}

define <2 x i64> @test_mask_andnot_epi64_rm_128(<2 x i64> %a, <2 x i64>* %ptr_b) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rm_128
  ;CHECK: vpandnq  (%rdi), %xmm0, %xmm0    ## encoding: [0x62,0xf1,0xfd,0x08,0xdf,0x07]
  %b = load <2 x i64>, <2 x i64>* %ptr_b
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> zeroinitializer, i8 -1)
  ret <2 x i64> %res
}

define <2 x i64> @test_mask_andnot_epi64_rmk_128(<2 x i64> %a, <2 x i64>* %ptr_b, <2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmk_128
  ;CHECK: vpandnq  (%rdi), %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x09,0xdf,0x0f]
  %b = load <2 x i64>, <2 x i64>* %ptr_b
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> %passThru, i8 %mask)
  ret <2 x i64> %res
}

define <2 x i64> @test_mask_andnot_epi64_rmkz_128(<2 x i64> %a, <2 x i64>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmkz_128
  ;CHECK: vpandnq  (%rdi), %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x89,0xdf,0x07]
  %b = load <2 x i64>, <2 x i64>* %ptr_b
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> zeroinitializer, i8 %mask)
  ret <2 x i64> %res
}

define <2 x i64> @test_mask_andnot_epi64_rmb_128(<2 x i64> %a, i64* %ptr_b) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmb_128
  ;CHECK: vpandnq  (%rdi){1to2}, %xmm0, %xmm0  ## encoding: [0x62,0xf1,0xfd,0x18,0xdf,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <2 x i64> undef, i64 %q, i32 0
  %b = shufflevector <2 x i64> %vecinit.i, <2 x i64> undef, <2 x i32> zeroinitializer
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> zeroinitializer, i8 -1)
  ret <2 x i64> %res
}

define <2 x i64> @test_mask_andnot_epi64_rmbk_128(<2 x i64> %a, i64* %ptr_b, <2 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmbk_128
  ;CHECK: vpandnq  (%rdi){1to2}, %xmm0, %xmm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x19,0xdf,0x0f]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <2 x i64> undef, i64 %q, i32 0
  %b = shufflevector <2 x i64> %vecinit.i, <2 x i64> undef, <2 x i32> zeroinitializer
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> %passThru, i8 %mask)
  ret <2 x i64> %res
}

define <2 x i64> @test_mask_andnot_epi64_rmbkz_128(<2 x i64> %a, i64* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmbkz_128
  ;CHECK: vpandnq  (%rdi){1to2}, %xmm0, %xmm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0x99,0xdf,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <2 x i64> undef, i64 %q, i32 0
  %b = shufflevector <2 x i64> %vecinit.i, <2 x i64> undef, <2 x i32> zeroinitializer
  %res = call <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64> %a, <2 x i64> %b, <2 x i64> zeroinitializer, i8 %mask)
  ret <2 x i64> %res
}

declare <2 x i64> @llvm.x86.avx512.mask.pandn.q.128(<2 x i64>, <2 x i64>, <2 x i64>, i8)

define <4 x i64> @test_mask_andnot_epi64_rr_256(<4 x i64> %a, <4 x i64> %b) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rr_256
  ;CHECK: vpandnq  %ymm1, %ymm0, %ymm0     ## encoding: [0x62,0xf1,0xfd,0x28,0xdf,0xc1]
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> zeroinitializer, i8 -1)
  ret <4 x i64> %res
}

define <4 x i64> @test_mask_andnot_epi64_rrk_256(<4 x i64> %a, <4 x i64> %b, <4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rrk_256
  ;CHECK: vpandnq  %ymm1, %ymm0, %ymm2 {%k1} ## encoding: [0x62,0xf1,0xfd,0x29,0xdf,0xd1]
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> %passThru, i8 %mask)
  ret <4 x i64> %res
}

define <4 x i64> @test_mask_andnot_epi64_rrkz_256(<4 x i64> %a, <4 x i64> %b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rrkz_256
  ;CHECK: vpandnq  %ymm1, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xa9,0xdf,0xc1]
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> zeroinitializer, i8 %mask)
  ret <4 x i64> %res
}

define <4 x i64> @test_mask_andnot_epi64_rm_256(<4 x i64> %a, <4 x i64>* %ptr_b) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rm_256
  ;CHECK: vpandnq  (%rdi), %ymm0, %ymm0    ## encoding: [0x62,0xf1,0xfd,0x28,0xdf,0x07]
  %b = load <4 x i64>, <4 x i64>* %ptr_b
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> zeroinitializer, i8 -1)
  ret <4 x i64> %res
}

define <4 x i64> @test_mask_andnot_epi64_rmk_256(<4 x i64> %a, <4 x i64>* %ptr_b, <4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmk_256
  ;CHECK: vpandnq  (%rdi), %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x29,0xdf,0x0f]
  %b = load <4 x i64>, <4 x i64>* %ptr_b
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> %passThru, i8 %mask)
  ret <4 x i64> %res
}

define <4 x i64> @test_mask_andnot_epi64_rmkz_256(<4 x i64> %a, <4 x i64>* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmkz_256
  ;CHECK: vpandnq  (%rdi), %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xa9,0xdf,0x07]
  %b = load <4 x i64>, <4 x i64>* %ptr_b
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> zeroinitializer, i8 %mask)
  ret <4 x i64> %res
}

define <4 x i64> @test_mask_andnot_epi64_rmb_256(<4 x i64> %a, i64* %ptr_b) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmb_256
  ;CHECK: vpandnq  (%rdi){1to4}, %ymm0, %ymm0  ## encoding: [0x62,0xf1,0xfd,0x38,0xdf,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <4 x i64> undef, i64 %q, i32 0
  %b = shufflevector <4 x i64> %vecinit.i, <4 x i64> undef, <4 x i32> zeroinitializer
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> zeroinitializer, i8 -1)
  ret <4 x i64> %res
}

define <4 x i64> @test_mask_andnot_epi64_rmbk_256(<4 x i64> %a, i64* %ptr_b, <4 x i64> %passThru, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmbk_256
  ;CHECK: vpandnq  (%rdi){1to4}, %ymm0, %ymm1 {%k1} ## encoding: [0x62,0xf1,0xfd,0x39,0xdf,0x0f]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <4 x i64> undef, i64 %q, i32 0
  %b = shufflevector <4 x i64> %vecinit.i, <4 x i64> undef, <4 x i32> zeroinitializer
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> %passThru, i8 %mask)
  ret <4 x i64> %res
}

define <4 x i64> @test_mask_andnot_epi64_rmbkz_256(<4 x i64> %a, i64* %ptr_b, i8 %mask) {
  ;CHECK-LABEL: test_mask_andnot_epi64_rmbkz_256
  ;CHECK: vpandnq  (%rdi){1to4}, %ymm0, %ymm0 {%k1} {z} ## encoding: [0x62,0xf1,0xfd,0xb9,0xdf,0x07]
  %q = load i64, i64* %ptr_b
  %vecinit.i = insertelement <4 x i64> undef, i64 %q, i32 0
  %b = shufflevector <4 x i64> %vecinit.i, <4 x i64> undef, <4 x i32> zeroinitializer
  %res = call <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64> %a, <4 x i64> %b, <4 x i64> zeroinitializer, i8 %mask)
  ret <4 x i64> %res
}

declare <4 x i64> @llvm.x86.avx512.mask.pandn.q.256(<4 x i64>, <4 x i64>, <4 x i64>, i8)

define i8 @test_cmpps_256(<8 x float> %a, <8 x float> %b) {
 ;CHECK: vcmpleps  %ymm1, %ymm0, %k0  ## encoding: [0x62,0xf1,0x7c,0x28,0xc2,0xc1,0x02]
   %res = call i8 @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> %a, <8 x float> %b, i32 2, i8 -1)
   ret i8 %res
 }
 declare i8 @llvm.x86.avx512.mask.cmp.ps.256(<8 x float> , <8 x float> , i32, i8)

define i8 @test_cmpps_128(<4 x float> %a, <4 x float> %b) {
 ;CHECK: vcmpleps  %xmm1, %xmm0, %k0  ## encoding: [0x62,0xf1,0x7c,0x08,0xc2,0xc1,0x02]
   %res = call i8 @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> %a, <4 x float> %b, i32 2, i8 -1)
   ret i8 %res
 }
 declare i8 @llvm.x86.avx512.mask.cmp.ps.128(<4 x float> , <4 x float> , i32, i8)

define i8 @test_cmppd_256(<4 x double> %a, <4 x double> %b) {
 ;CHECK: vcmplepd  %ymm1, %ymm0, %k0  ## encoding: [0x62,0xf1,0xfd,0x28,0xc2,0xc1,0x02]
   %res = call i8 @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> %a, <4 x double> %b, i32 2, i8 -1)
   ret i8 %res
 }
 declare i8 @llvm.x86.avx512.mask.cmp.pd.256(<4 x double> , <4 x double> , i32, i8)

define i8 @test_cmppd_128(<2 x double> %a, <2 x double> %b) {
 ;CHECK: vcmplepd  %xmm1, %xmm0, %k0  ## encoding: [0x62,0xf1,0xfd,0x08,0xc2,0xc1,0x02]
   %res = call i8 @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> %a, <2 x double> %b, i32 2, i8 -1)
   ret i8 %res
 }
 declare i8 @llvm.x86.avx512.mask.cmp.pd.128(<2 x double> , <2 x double> , i32, i8)

define <8 x float> @test_mm512_maskz_add_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_add_ps_256
  ;CHECK: vaddps %ymm1, %ymm0, %ymm0 {%k1} {z}
  %res = call <8 x float> @llvm.x86.avx512.mask.add.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_mask_add_ps_256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_add_ps_256
  ;CHECK: vaddps %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <8 x float> @llvm.x86.avx512.mask.add.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_add_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_add_ps_256
  ;CHECK: vaddps %ymm1, %ymm0, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.mask.add.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 -1)
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx512.mask.add.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <4 x float> @test_mm512_maskz_add_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_add_ps_128
  ;CHECK: vaddps %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.mask.add.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_mask_add_ps_128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_add_ps_128
  ;CHECK: vaddps %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <4 x float> @llvm.x86.avx512.mask.add.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_add_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_add_ps_128
  ;CHECK: vaddps %xmm1, %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.add.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 -1)
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.mask.add.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <8 x float> @test_mm512_maskz_sub_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_sub_ps_256
  ;CHECK: vsubps %ymm1, %ymm0, %ymm0 {%k1} {z}
  %res = call <8 x float> @llvm.x86.avx512.mask.sub.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_mask_sub_ps_256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_sub_ps_256
  ;CHECK: vsubps %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <8 x float> @llvm.x86.avx512.mask.sub.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_sub_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_sub_ps_256
  ;CHECK: vsubps %ymm1, %ymm0, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.mask.sub.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 -1)
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx512.mask.sub.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <4 x float> @test_mm512_maskz_sub_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_sub_ps_128
  ;CHECK: vsubps %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.mask.sub.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_mask_sub_ps_128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_sub_ps_128
  ;CHECK: vsubps %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <4 x float> @llvm.x86.avx512.mask.sub.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_sub_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_sub_ps_128
  ;CHECK: vsubps %xmm1, %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.sub.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 -1)
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.mask.sub.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <8 x float> @test_mm512_maskz_mul_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_mul_ps_256
  ;CHECK: vmulps %ymm1, %ymm0, %ymm0 {%k1} {z}
  %res = call <8 x float> @llvm.x86.avx512.mask.mul.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_mask_mul_ps_256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_mul_ps_256
  ;CHECK: vmulps %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <8 x float> @llvm.x86.avx512.mask.mul.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_mul_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mul_ps_256
  ;CHECK: vmulps %ymm1, %ymm0, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.mask.mul.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 -1)
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx512.mask.mul.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <4 x float> @test_mm512_maskz_mul_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_mul_ps_128
  ;CHECK: vmulps %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.mask.mul.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_mask_mul_ps_128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_mul_ps_128
  ;CHECK: vmulps %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <4 x float> @llvm.x86.avx512.mask.mul.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_mul_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mul_ps_128
  ;CHECK: vmulps %xmm1, %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.mul.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 -1)
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.mask.mul.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <8 x float> @test_mm512_maskz_div_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_div_ps_256
  ;CHECK: vdivps %ymm1, %ymm0, %ymm0 {%k1} {z}
  %res = call <8 x float> @llvm.x86.avx512.mask.div.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_mask_div_ps_256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_div_ps_256
  ;CHECK: vdivps %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <8 x float> @llvm.x86.avx512.mask.div.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_div_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_div_ps_256
  ;CHECK: vdivps %ymm1, %ymm0, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.mask.div.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 -1)
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx512.mask.div.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <4 x float> @test_mm512_maskz_div_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_div_ps_128
  ;CHECK: vdivps %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.mask.div.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_mask_div_ps_128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_div_ps_128
  ;CHECK: vdivps %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <4 x float> @llvm.x86.avx512.mask.div.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_div_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_div_ps_128
  ;CHECK: vdivps %xmm1, %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.div.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 -1)
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.mask.div.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <8 x float> @test_mm512_maskz_max_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_max_ps_256
  ;CHECK: vmaxps %ymm1, %ymm0, %ymm0 {%k1} {z}
  %res = call <8 x float> @llvm.x86.avx512.mask.max.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_mask_max_ps_256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_max_ps_256
  ;CHECK: vmaxps %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <8 x float> @llvm.x86.avx512.mask.max.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_max_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_max_ps_256
  ;CHECK: vmaxps %ymm1, %ymm0, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.mask.max.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 -1)
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx512.mask.max.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <4 x float> @test_mm512_maskz_max_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_max_ps_128
  ;CHECK: vmaxps %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.mask.max.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_mask_max_ps_128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_max_ps_128
  ;CHECK: vmaxps %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <4 x float> @llvm.x86.avx512.mask.max.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_max_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_max_ps_128
  ;CHECK: vmaxps %xmm1, %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.max.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 -1)
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.mask.max.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)

define <8 x float> @test_mm512_maskz_min_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_min_ps_256
  ;CHECK: vminps %ymm1, %ymm0, %ymm0 {%k1} {z}
  %res = call <8 x float> @llvm.x86.avx512.mask.min.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_mask_min_ps_256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_min_ps_256
  ;CHECK: vminps %ymm1, %ymm0, %ymm2 {%k1} 
  %res = call <8 x float> @llvm.x86.avx512.mask.min.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %src, i8 %mask)
  ret <8 x float> %res
}

define <8 x float> @test_mm512_min_ps_256(<8 x float> %a0, <8 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_min_ps_256
  ;CHECK: vminps %ymm1, %ymm0, %ymm0
  %res = call <8 x float> @llvm.x86.avx512.mask.min.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float>zeroinitializer, i8 -1)
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx512.mask.min.ps.256(<8 x float>, <8 x float>, <8 x float>, i8)

define <4 x float> @test_mm512_maskz_min_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_maskz_min_ps_128
  ;CHECK: vminps %xmm1, %xmm0, %xmm0 {%k1} {z}
  %res = call <4 x float> @llvm.x86.avx512.mask.min.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_mask_min_ps_128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask) {
  ;CHECK-LABEL: test_mm512_mask_min_ps_128
  ;CHECK: vminps %xmm1, %xmm0, %xmm2 {%k1} 
  %res = call <4 x float> @llvm.x86.avx512.mask.min.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float> %src, i8 %mask)
  ret <4 x float> %res
}

define <4 x float> @test_mm512_min_ps_128(<4 x float> %a0, <4 x float> %a1, i8 %mask) {
  ;CHECK-LABEL: test_mm512_min_ps_128
  ;CHECK: vminps %xmm1, %xmm0, %xmm0
  %res = call <4 x float> @llvm.x86.avx512.mask.min.ps.128(<4 x float> %a0, <4 x float> %a1, <4 x float>zeroinitializer, i8 -1)
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.mask.min.ps.128(<4 x float>, <4 x float>, <4 x float>, i8)
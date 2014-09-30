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

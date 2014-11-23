; RUN: llc -mtriple=x86_64-apple-darwin  -mcpu=knl < %s | FileCheck %s -check-prefix=AVX512
; RUN: llc -mtriple=x86_64-apple-darwin  -mcpu=core-avx2 < %s | FileCheck %s -check-prefix=AVX2

; AVX512-LABEL: test1
; AVX512: vmovdqu32       (%rdi), %zmm0 {%k1} {z}

; AVX2-LABEL: test1
; AVX2: vpmaskmovd      32(%rdi)
; AVX2: vpmaskmovd      (%rdi)
; AVX2-NOT: blend

define <16 x i32> @test1(<16 x i32> %trigger, i8* %addr) {
  %mask = icmp eq <16 x i32> %trigger, zeroinitializer
  %res = call <16 x i32> @llvm.masked.load.v16i32(i8* %addr, <16 x i32>undef, i32 4, <16 x i1>%mask)
  ret <16 x i32> %res
}

; AVX512-LABEL: test2
; AVX512: vmovdqu32       (%rdi), %zmm0 {%k1} {z}

; AVX2-LABEL: test2
; AVX2: vpmaskmovd      {{.*}}(%rdi)
; AVX2: vpmaskmovd      {{.*}}(%rdi)
; AVX2-NOT: blend
define <16 x i32> @test2(<16 x i32> %trigger, i8* %addr) {
  %mask = icmp eq <16 x i32> %trigger, zeroinitializer
  %res = call <16 x i32> @llvm.masked.load.v16i32(i8* %addr, <16 x i32>zeroinitializer, i32 4, <16 x i1>%mask)
  ret <16 x i32> %res
}

; AVX512-LABEL: test3
; AVX512: vmovdqu32       %zmm1, (%rdi) {%k1}

define void @test3(<16 x i32> %trigger, i8* %addr, <16 x i32> %val) {
  %mask = icmp eq <16 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v16i32(i8* %addr, <16 x i32>%val, i32 4, <16 x i1>%mask)
  ret void
}

; AVX512-LABEL: test4
; AVX512: vmovups       (%rdi), %zmm{{.*{%k[1-7]}}}

; AVX2-LABEL: test4
; AVX2: vpmaskmovd      {{.*}}(%rdi)
; AVX2: vpmaskmovd      {{.*}}(%rdi)
; AVX2: blend
define <16 x float> @test4(<16 x i32> %trigger, i8* %addr, <16 x float> %dst) {
  %mask = icmp eq <16 x i32> %trigger, zeroinitializer
  %res = call <16 x float> @llvm.masked.load.v16f32(i8* %addr, <16 x float>%dst, i32 4, <16 x i1>%mask)
  ret <16 x float> %res
}

; AVX512-LABEL: test5
; AVX512: vmovupd (%rdi), %zmm1 {%k1}

; AVX2-LABEL: test5
; AVX2: vpmaskmovq
; AVX2: vblendvpd
; AVX2: vpmaskmovq  
; AVX2: vblendvpd
define <8 x double> @test5(<8 x i32> %trigger, i8* %addr, <8 x double> %dst) {
  %mask = icmp eq <8 x i32> %trigger, zeroinitializer
  %res = call <8 x double> @llvm.masked.load.v8f64(i8* %addr, <8 x double>%dst, i32 4, <8 x i1>%mask)
  ret <8 x double> %res
}

declare <16 x i32> @llvm.masked.load.v16i32(i8*, <16 x i32>, i32, <16 x i1>) 
declare void @llvm.masked.store.v16i32(i8*, <16 x i32>, i32, <16 x i1>) 
declare <16 x float> @llvm.masked.load.v16f32(i8*, <16 x float>, i32, <16 x i1>) 
declare void @llvm.masked.store.v16f32(i8*, <16 x float>, i32, <16 x i1>) 
declare <8 x double> @llvm.masked.load.v8f64(i8*, <8 x double>, i32, <8 x i1>) 
declare void @llvm.masked.store.v8f64(i8*, <8 x double>, i32, <8 x i1>) 


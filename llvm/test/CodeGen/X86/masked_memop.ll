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
; AVX2: vmaskmovps      {{.*}}(%rdi)
; AVX2: vmaskmovps      {{.*}}(%rdi)
; AVX2: blend
define <16 x float> @test4(<16 x i32> %trigger, i8* %addr, <16 x float> %dst) {
  %mask = icmp eq <16 x i32> %trigger, zeroinitializer
  %res = call <16 x float> @llvm.masked.load.v16f32(i8* %addr, <16 x float>%dst, i32 4, <16 x i1>%mask)
  ret <16 x float> %res
}

; AVX512-LABEL: test5
; AVX512: vmovupd (%rdi), %zmm1 {%k1}

; AVX2-LABEL: test5
; AVX2: vmaskmovpd
; AVX2: vblendvpd
; AVX2: vmaskmovpd  
; AVX2: vblendvpd
define <8 x double> @test5(<8 x i32> %trigger, i8* %addr, <8 x double> %dst) {
  %mask = icmp eq <8 x i32> %trigger, zeroinitializer
  %res = call <8 x double> @llvm.masked.load.v8f64(i8* %addr, <8 x double>%dst, i32 4, <8 x i1>%mask)
  ret <8 x double> %res
}

; AVX2-LABEL: test6
; AVX2: vmaskmovpd
; AVX2: vblendvpd
define <2 x double> @test6(<2 x i64> %trigger, i8* %addr, <2 x double> %dst) {
  %mask = icmp eq <2 x i64> %trigger, zeroinitializer
  %res = call <2 x double> @llvm.masked.load.v2f64(i8* %addr, <2 x double>%dst, i32 4, <2 x i1>%mask)
  ret <2 x double> %res
}

; AVX2-LABEL: test7
; AVX2: vmaskmovps      {{.*}}(%rdi)
; AVX2: blend
define <4 x float> @test7(<4 x i32> %trigger, i8* %addr, <4 x float> %dst) {
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  %res = call <4 x float> @llvm.masked.load.v4f32(i8* %addr, <4 x float>%dst, i32 4, <4 x i1>%mask)
  ret <4 x float> %res
}

; AVX2-LABEL: test8
; AVX2: vpmaskmovd      {{.*}}(%rdi)
; AVX2: blend
define <4 x i32> @test8(<4 x i32> %trigger, i8* %addr, <4 x i32> %dst) {
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  %res = call <4 x i32> @llvm.masked.load.v4i32(i8* %addr, <4 x i32>%dst, i32 4, <4 x i1>%mask)
  ret <4 x i32> %res
}

; AVX2-LABEL: test9
; AVX2: vpmaskmovd %xmm
define void @test9(<4 x i32> %trigger, i8* %addr, <4 x i32> %val) {
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v4i32(i8* %addr, <4 x i32>%val, i32 4, <4 x i1>%mask)
  ret void
}

; AVX2-LABEL: test10
; AVX2: vmaskmovpd    (%rdi), %ymm
; AVX2: blend
define <4 x double> @test10(<4 x i32> %trigger, i8* %addr, <4 x double> %dst) {
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  %res = call <4 x double> @llvm.masked.load.v4f64(i8* %addr, <4 x double>%dst, i32 4, <4 x i1>%mask)
  ret <4 x double> %res
}

; AVX2-LABEL: test11
; AVX2: vmaskmovps
; AVX2: vblendvps
define <8 x float> @test11(<8 x i32> %trigger, i8* %addr, <8 x float> %dst) {
  %mask = icmp eq <8 x i32> %trigger, zeroinitializer
  %res = call <8 x float> @llvm.masked.load.v8f32(i8* %addr, <8 x float>%dst, i32 4, <8 x i1>%mask)
  ret <8 x float> %res
}

; AVX2-LABEL: test12
; AVX2: vpmaskmovd %ymm
define void @test12(<8 x i32> %trigger, i8* %addr, <8 x i32> %val) {
  %mask = icmp eq <8 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v8i32(i8* %addr, <8 x i32>%val, i32 4, <8 x i1>%mask)
  ret void
}

declare <16 x i32> @llvm.masked.load.v16i32(i8*, <16 x i32>, i32, <16 x i1>) 
declare <4 x i32> @llvm.masked.load.v4i32(i8*, <4 x i32>, i32, <4 x i1>)
declare void @llvm.masked.store.v16i32(i8*, <16 x i32>, i32, <16 x i1>)
declare void @llvm.masked.store.v8i32(i8*, <8 x i32>, i32, <8 x i1>)
declare void @llvm.masked.store.v4i32(i8*, <4 x i32>, i32, <4 x i1>)
declare <16 x float> @llvm.masked.load.v16f32(i8*, <16 x float>, i32, <16 x i1>)
declare <8 x float> @llvm.masked.load.v8f32(i8*, <8 x float>, i32, <8 x i1>)
declare <4 x float> @llvm.masked.load.v4f32(i8*, <4 x float>, i32, <4 x i1>)
declare void @llvm.masked.store.v16f32(i8*, <16 x float>, i32, <16 x i1>) 
declare <8 x double> @llvm.masked.load.v8f64(i8*, <8 x double>, i32, <8 x i1>)
declare <4 x double> @llvm.masked.load.v4f64(i8*, <4 x double>, i32, <4 x i1>)
declare <2 x double> @llvm.masked.load.v2f64(i8*, <2 x double>, i32, <2 x i1>)
declare void @llvm.masked.store.v8f64(i8*, <8 x double>, i32, <8 x i1>)
declare void @llvm.masked.store.v2f64(i8*, <2 x double>, i32, <2 x i1>)
declare void @llvm.masked.store.v2i64(i8*, <2 x i64>, i32, <2 x i1>)


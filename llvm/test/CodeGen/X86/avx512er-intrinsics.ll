; RUN: llc < %s -mtriple=x86_64-pc-linux -mcpu=knl --show-mc-encoding| FileCheck %s

define <16 x float> @test_rsqrt28_ps(<16 x float> %a0) {
  ; CHECK: vrsqrt28ps %zmm0, %zmm0 {sae} # encoding: [0x62,0xf2,0x7d,0x18,0xcc,0xc0]
  %res = call <16 x float> @llvm.x86.avx512.rsqrt28.ps(<16 x float> %a0, <16 x float> zeroinitializer, i16 -1, i32 8)
  ret <16 x float> %res
}

define <16 x float> @test1_rsqrt28_ps(<16 x float> %a0, <16 x float> %a1) {
  ; CHECK: kmovw
  ; CHECK: vrsqrt28ps %zmm0, %zmm1 {%k1}{sae} # encoding: [0x62,0xf2,0x7d,0x19,0xcc,0xc8]
  %res = call <16 x float> @llvm.x86.avx512.rsqrt28.ps(<16 x float> %a0, <16 x float> %a1, i16 6, i32 8)
  ret <16 x float> %res
}

define <16 x float> @test2_rsqrt28_ps(<16 x float> %a0) {
  ; CHECK: vrsqrt28ps %zmm0, %zmm0 {%k1} {z} # encoding: [0x62,0xf2,0x7d,0xc9,0xcc,0xc0]
  %res = call <16 x float> @llvm.x86.avx512.rsqrt28.ps(<16 x float> %a0, <16 x float> undef, i16 6, i32 4)
  ret <16 x float> %res
}

define <16 x float> @test3_rsqrt28_ps(<16 x float> %a0) {
  ; CHECK: kmovw
  ; CHECK: vrsqrt28ps %zmm0, %zmm0 {%k1} {z} # encoding: [0x62,0xf2,0x7d,0xc9,0xcc,0xc0]
  %res = call <16 x float> @llvm.x86.avx512.rsqrt28.ps(<16 x float> %a0, <16 x float> zeroinitializer, i16 6, i32 4)
  ret <16 x float> %res
}

define <16 x float> @test4_rsqrt28_ps(<16 x float> %a0) {
  ; CHECK: vrsqrt28ps %zmm0, %zmm0 {%k1} {z}{sae} # encoding: [0x62,0xf2,0x7d,0x99,0xcc,0xc0]
  %res = call <16 x float> @llvm.x86.avx512.rsqrt28.ps(<16 x float> %a0, <16 x float> undef, i16 6, i32 8) 
  ret <16 x float> %res
}


declare <16 x float> @llvm.x86.avx512.rsqrt28.ps(<16 x float>, <16 x float>, i16, i32) nounwind readnone

define <16 x float> @test_rcp28_ps_512(<16 x float> %a0) {
  ; CHECK: vrcp28ps %zmm0, %zmm0 {sae} # encoding: [0x62,0xf2,0x7d,0x18,0xca,0xc0]
  %res = call <16 x float> @llvm.x86.avx512.rcp28.ps(<16 x float> %a0, <16 x float> zeroinitializer, i16 -1, i32 8)
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.rcp28.ps(<16 x float>, <16 x float>, i16, i32) nounwind readnone

define <8 x double> @test_rcp28_pd_512(<8 x double> %a0) {
  ; CHECK: vrcp28pd %zmm0, %zmm0 {sae} # encoding: [0x62,0xf2,0xfd,0x18,0xca,0xc0]
  %res = call <8 x double> @llvm.x86.avx512.rcp28.pd(<8 x double> %a0, <8 x double> zeroinitializer, i8 -1, i32 8) 
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.rcp28.pd(<8 x double>, <8 x double>, i8, i32) nounwind readnone

define <16 x float> @test_exp2_ps_512(<16 x float> %a0) {
  ; CHECK: vexp2ps %zmm0, %zmm0 {sae}      # encoding: [0x62,0xf2,0x7d,0x18,0xc8,0xc0]
  %res = call <16 x float> @llvm.x86.avx512.exp2.ps(<16 x float> %a0, <16 x float> zeroinitializer, i16 -1, i32 8)
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.exp2.ps(<16 x float>, <16 x float>, i16, i32) nounwind readnone

define <8 x double> @test_exp2_pd_512(<8 x double> %a0) {
  ; CHECK: vexp2pd %zmm0, %zmm0 {sae}      # encoding: [0x62,0xf2,0xfd,0x18,0xc8,0xc0]
  %res = call <8 x double> @llvm.x86.avx512.exp2.pd(<8 x double> %a0, <8 x double> zeroinitializer, i8 -1, i32 8)
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.exp2.pd(<8 x double>, <8 x double>, i8, i32) nounwind readnone

define <4 x float> @test_rsqrt28_ss(<4 x float> %a0) {
  ; CHECK: vrsqrt28ss %xmm0, %xmm0, %xmm0 {sae} # encoding: [0x62,0xf2,0x7d,0x18,0xcd,0xc0]
  %res = call <4 x float> @llvm.x86.avx512.rsqrt28.ss(<4 x float> %a0, <4 x float> %a0, <4 x float> zeroinitializer, i8 -1, i32 8) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.rsqrt28.ss(<4 x float>, <4 x float>, <4 x float>, i8, i32) nounwind readnone

define <4 x float> @test_rcp28_ss(<4 x float> %a0) {
  ; CHECK: vrcp28ss %xmm0, %xmm0, %xmm0 {sae} # encoding: [0x62,0xf2,0x7d,0x18,0xcb,0xc0]
  %res = call <4 x float> @llvm.x86.avx512.rcp28.ss(<4 x float> %a0, <4 x float> %a0, <4 x float> zeroinitializer, i8 -1, i32 8) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.rcp28.ss(<4 x float>, <4 x float>, <4 x float>, i8, i32) nounwind readnone

define <4 x float> @test_rsqrt28_ss_maskz(<4 x float> %a0) {
  ; CHECK: vrsqrt28ss %xmm0, %xmm0, %xmm0 {%k1} {z}{sae} # encoding: [0x62,0xf2,0x7d,0x99,0xcd,0xc0]
  %res = call <4 x float> @llvm.x86.avx512.rsqrt28.ss(<4 x float> %a0, <4 x float> %a0, <4 x float> zeroinitializer, i8 7, i32 8) ; 
  ret <4 x float> %res
}

define <4 x float> @test_rsqrt28_ss_mask(<4 x float> %a0, <4 x float> %b0, <4 x float> %c0) {
  ; CHECK: vrsqrt28ss %xmm1, %xmm0, %xmm2 {%k1}{sae} # encoding: [0x62,0xf2,0x7d,0x19,0xcd,0xd1]
  %res = call <4 x float> @llvm.x86.avx512.rsqrt28.ss(<4 x float> %a0, <4 x float> %b0, <4 x float> %c0, i8 7, i32 8) ;
  ret <4 x float> %res
}

define <2 x double> @test_rsqrt28_sd_maskz(<2 x double> %a0) {
  ; CHECK: vrsqrt28sd %xmm0, %xmm0, %xmm0 {%k1} {z}{sae} # encoding: [0x62,0xf2,0xfd,0x99,0xcd,0xc0]
  %res = call <2 x double> @llvm.x86.avx512.rsqrt28.sd(<2 x double> %a0, <2 x double> %a0, <2 x double> zeroinitializer, i8 7, i32 8) ; 
  ret <2 x double> %res
}

declare <2 x double> @llvm.x86.avx512.rsqrt28.sd(<2 x double>, <2 x double>, <2 x double>, i8, i32) nounwind readnone

define <2 x double> @test_rsqrt28_sd_maskz_mem(<2 x double> %a0, double* %ptr ) {
  ; CHECK: vrsqrt28sd (%rdi), %xmm0, %xmm0 {%k1} {z} # encoding: [0x62,0xf2,0xfd,0x89,0xcd,0x07]
  %mem = load double , double * %ptr, align 8
  %mem_v = insertelement <2 x double> undef, double %mem, i32 0
  %res = call <2 x double> @llvm.x86.avx512.rsqrt28.sd(<2 x double> %a0, <2 x double> %mem_v, <2 x double> zeroinitializer, i8 7, i32 4) ; 
  ret <2 x double> %res
}

define <2 x double> @test_rsqrt28_sd_maskz_mem_offset(<2 x double> %a0, double* %ptr ) {
  ; CHECK: vrsqrt28sd 144(%rdi), %xmm0, %xmm0 {%k1} {z} # encoding: [0x62,0xf2,0xfd,0x89,0xcd,0x47,0x12]
  %ptr1 = getelementptr double, double* %ptr, i32 18
  %mem = load double , double * %ptr1, align 8
  %mem_v = insertelement <2 x double> undef, double %mem, i32 0
  %res = call <2 x double> @llvm.x86.avx512.rsqrt28.sd(<2 x double> %a0, <2 x double> %mem_v, <2 x double> zeroinitializer, i8 7, i32 4) ;
  ret <2 x double> %res
}


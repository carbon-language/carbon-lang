; RUN: llc < %s -mtriple=x86_64-unknown-unknown -march=x86-64 -mcpu=corei7-avx -mattr=+fma | FileCheck %s --check-prefix=CHECK-FMA --check-prefix=CHECK
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -march=x86-64 -mcpu=core-avx2 -mattr=+fma,+avx2 | FileCheck %s --check-prefix=CHECK-FMA --check-prefix=CHECK
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -march=x86-64 -mcpu=corei7-avx -mattr=+fma4 | FileCheck %s --check-prefix=CHECK-FMA4 --check-prefix=CHECK
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=bdver2 -mattr=+avx,-fma | FileCheck %s --check-prefix=CHECK-FMA4 --check-prefix=CHECK

; VFMADD
define < 4 x float > @test_x86_fma_vfmadd_ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK-FMA4: vfmaddss
  ; CHECK-FMA: vfmadd213ss
  %res = call < 4 x float > @llvm.x86.fma.vfmadd.ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2)
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfmadd.ss(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfmadd_sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK-FMA4: vfmaddsd
  ; CHECK-FMA: vfmadd213sd
  %res = call < 2 x double > @llvm.x86.fma.vfmadd.sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2)
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfmadd.sd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 4 x float > @test_x86_fma_vfmadd_ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK-FMA4: vfmaddps
  ; CHECK-FMA: vfmadd213ps
  %res = call < 4 x float > @llvm.x86.fma.vfmadd.ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2)
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfmadd.ps(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfmadd_pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK-FMA4: vfmaddpd
  ; CHECK-FMA: vfmadd213pd
  %res = call < 2 x double > @llvm.x86.fma.vfmadd.pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2)
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfmadd.pd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 8 x float > @test_x86_fma_vfmadd_ps_256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) {
  ; CHECK-FMA4: vfmaddps
  ; CHECK-FMA: vfmadd213ps
  ; CHECK: ymm
  %res = call < 8 x float > @llvm.x86.fma.vfmadd.ps.256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2)
  ret < 8 x float > %res
}
declare < 8 x float > @llvm.x86.fma.vfmadd.ps.256(< 8 x float >, < 8 x float >, < 8 x float >) nounwind readnone

define < 4 x double > @test_x86_fma_vfmadd_pd_256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) {
  ; CHECK-FMA4: vfmaddpd
  ; CHECK-FMA: vfmadd213pd
  ; CHECK: ymm
  %res = call < 4 x double > @llvm.x86.fma.vfmadd.pd.256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2)
  ret < 4 x double > %res
}
declare < 4 x double > @llvm.x86.fma.vfmadd.pd.256(< 4 x double >, < 4 x double >, < 4 x double >) nounwind readnone

; VFMSUB
define < 4 x float > @test_x86_fma_vfmsub_ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK-FMA4: vfmsubss
  ; CHECK-FMA: vfmsub213ss
  %res = call < 4 x float > @llvm.x86.fma.vfmsub.ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2)
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfmsub.ss(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfmsub_sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK-FMA4: vfmsubsd
  ; CHECK-FMA: vfmsub213sd
  %res = call < 2 x double > @llvm.x86.fma.vfmsub.sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2)
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfmsub.sd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 4 x float > @test_x86_fma_vfmsub_ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK-FMA4: vfmsubps
  ; CHECK-FMA: vfmsub213ps
  %res = call < 4 x float > @llvm.x86.fma.vfmsub.ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2)
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfmsub.ps(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfmsub_pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK-FMA4: vfmsubpd
  ; CHECK-FMA: vfmsub213pd
  %res = call < 2 x double > @llvm.x86.fma.vfmsub.pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2)
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfmsub.pd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 8 x float > @test_x86_fma_vfmsub_ps_256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) {
  ; CHECK-FMA4: vfmsubps
  ; CHECK-FMA: vfmsub213ps
  ; CHECK: ymm
  %res = call < 8 x float > @llvm.x86.fma.vfmsub.ps.256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2)
  ret < 8 x float > %res
}
declare < 8 x float > @llvm.x86.fma.vfmsub.ps.256(< 8 x float >, < 8 x float >, < 8 x float >) nounwind readnone

define < 4 x double > @test_x86_fma_vfmsub_pd_256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) {
  ; CHECK-FMA4: vfmsubpd
  ; CHECK-FMA: vfmsub213pd
  ; CHECK: ymm
  %res = call < 4 x double > @llvm.x86.fma.vfmsub.pd.256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2)
  ret < 4 x double > %res
}
declare < 4 x double > @llvm.x86.fma.vfmsub.pd.256(< 4 x double >, < 4 x double >, < 4 x double >) nounwind readnone

; VFNMADD
define < 4 x float > @test_x86_fma_vfnmadd_ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK-FMA4: vfnmaddss
  ; CHECK-FMA: vfnmadd213ss
  %res = call < 4 x float > @llvm.x86.fma.vfnmadd.ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2)
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfnmadd.ss(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfnmadd_sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK-FMA4: vfnmaddsd
  ; CHECK-FMA: vfnmadd213sd
  %res = call < 2 x double > @llvm.x86.fma.vfnmadd.sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2)
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfnmadd.sd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 4 x float > @test_x86_fma_vfnmadd_ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK-FMA4: vfnmaddps
  ; CHECK-FMA: vfnmadd213ps
  %res = call < 4 x float > @llvm.x86.fma.vfnmadd.ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2)
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfnmadd.ps(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfnmadd_pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK-FMA4: vfnmaddpd
  ; CHECK-FMA: vfnmadd213pd
  %res = call < 2 x double > @llvm.x86.fma.vfnmadd.pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2)
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfnmadd.pd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 8 x float > @test_x86_fma_vfnmadd_ps_256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) {
  ; CHECK-FMA4: vfnmaddps
  ; CHECK-FMA: vfnmadd213ps
  ; CHECK: ymm
  %res = call < 8 x float > @llvm.x86.fma.vfnmadd.ps.256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2)
  ret < 8 x float > %res
}
declare < 8 x float > @llvm.x86.fma.vfnmadd.ps.256(< 8 x float >, < 8 x float >, < 8 x float >) nounwind readnone

define < 4 x double > @test_x86_fma_vfnmadd_pd_256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) {
  ; CHECK-FMA4: vfnmaddpd
  ; CHECK-FMA: vfnmadd213pd
  ; CHECK: ymm
  %res = call < 4 x double > @llvm.x86.fma.vfnmadd.pd.256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2)
  ret < 4 x double > %res
}
declare < 4 x double > @llvm.x86.fma.vfnmadd.pd.256(< 4 x double >, < 4 x double >, < 4 x double >) nounwind readnone

; VFNMSUB
define < 4 x float > @test_x86_fma_vfnmsub_ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK-FMA4: vfnmsubss
  ; CHECK-FMA: vfnmsub213ss
  %res = call < 4 x float > @llvm.x86.fma.vfnmsub.ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2)
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfnmsub.ss(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfnmsub_sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK-FMA4: vfnmsubsd
  ; CHECK-FMA: vfnmsub213sd
  %res = call < 2 x double > @llvm.x86.fma.vfnmsub.sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2)
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfnmsub.sd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 4 x float > @test_x86_fma_vfnmsub_ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK-FMA4: vfnmsubps
  ; CHECK-FMA: vfnmsub213ps
  %res = call < 4 x float > @llvm.x86.fma.vfnmsub.ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2)
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfnmsub.ps(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfnmsub_pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK-FMA4: vfnmsubpd
  ; CHECK-FMA: vfnmsub213pd
  %res = call < 2 x double > @llvm.x86.fma.vfnmsub.pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2)
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfnmsub.pd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 8 x float > @test_x86_fma_vfnmsub_ps_256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) {
  ; CHECK-FMA4: vfnmsubps
  ; CHECK-FMA: vfnmsub213ps
  ; CHECK: ymm
  %res = call < 8 x float > @llvm.x86.fma.vfnmsub.ps.256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2)
  ret < 8 x float > %res
}
declare < 8 x float > @llvm.x86.fma.vfnmsub.ps.256(< 8 x float >, < 8 x float >, < 8 x float >) nounwind readnone

define < 4 x double > @test_x86_fma_vfnmsub_pd_256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) {
  ; CHECK-FMA4: vfnmsubpd
  ; CHECK-FMA: vfnmsub213pd
  ; CHECK: ymm
  %res = call < 4 x double > @llvm.x86.fma.vfnmsub.pd.256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2)
  ret < 4 x double > %res
}
declare < 4 x double > @llvm.x86.fma.vfnmsub.pd.256(< 4 x double >, < 4 x double >, < 4 x double >) nounwind readnone

; VFMADDSUB
define < 4 x float > @test_x86_fma_vfmaddsub_ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK-FMA4: vfmaddsubps
  ; CHECK-FMA: vfmaddsub213ps
  %res = call < 4 x float > @llvm.x86.fma.vfmaddsub.ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2)
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfmaddsub.ps(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfmaddsub_pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK-FMA4: vfmaddsubpd
  ; CHECK-FMA: vfmaddsub213pd
  %res = call < 2 x double > @llvm.x86.fma.vfmaddsub.pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2)
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfmaddsub.pd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 8 x float > @test_x86_fma_vfmaddsub_ps_256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) {
  ; CHECK-FMA4: vfmaddsubps
  ; CHECK-FMA: vfmaddsub213ps
  ; CHECK: ymm
  %res = call < 8 x float > @llvm.x86.fma.vfmaddsub.ps.256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2)
  ret < 8 x float > %res
}
declare < 8 x float > @llvm.x86.fma.vfmaddsub.ps.256(< 8 x float >, < 8 x float >, < 8 x float >) nounwind readnone

define < 4 x double > @test_x86_fma_vfmaddsub_pd_256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) {
  ; CHECK-FMA4: vfmaddsubpd
  ; CHECK-FMA: vfmaddsub213pd
  ; CHECK: ymm
  %res = call < 4 x double > @llvm.x86.fma.vfmaddsub.pd.256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2)
  ret < 4 x double > %res
}
declare < 4 x double > @llvm.x86.fma.vfmaddsub.pd.256(< 4 x double >, < 4 x double >, < 4 x double >) nounwind readnone

; VFMSUBADD
define < 4 x float > @test_x86_fma_vfmsubadd_ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK-FMA4: vfmsubaddps
  ; CHECK-FMA: vfmsubadd213ps
  %res = call < 4 x float > @llvm.x86.fma.vfmsubadd.ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2)
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfmsubadd.ps(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfmsubadd_pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK-FMA4: vfmsubaddpd
  ; CHECK-FMA: vfmsubadd213pd
  %res = call < 2 x double > @llvm.x86.fma.vfmsubadd.pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2)
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfmsubadd.pd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 8 x float > @test_x86_fma_vfmsubadd_ps_256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) {
  ; CHECK-FMA4: vfmsubaddps
  ; CHECK-FMA: vfmsubadd213ps
  ; CHECK: ymm
  %res = call < 8 x float > @llvm.x86.fma.vfmsubadd.ps.256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2)
  ret < 8 x float > %res
}
declare < 8 x float > @llvm.x86.fma.vfmsubadd.ps.256(< 8 x float >, < 8 x float >, < 8 x float >) nounwind readnone

define < 4 x double > @test_x86_fma_vfmsubadd_pd_256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) {
  ; CHECK-FMA4: vfmsubaddpd
  ; CHECK-FMA: vfmsubadd213pd
  ; CHECK: ymm
  %res = call < 4 x double > @llvm.x86.fma.vfmsubadd.pd.256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2)
  ret < 4 x double > %res
}
declare < 4 x double > @llvm.x86.fma.vfmsubadd.pd.256(< 4 x double >, < 4 x double >, < 4 x double >) nounwind readnone

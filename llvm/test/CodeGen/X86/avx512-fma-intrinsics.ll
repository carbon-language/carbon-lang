; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

define <16 x float> @test_x86_vfmadd_ps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_x86_vfmadd_ps_z
  ; CHECK: vfmadd213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.vfmadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) nounwind
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.fma.vfmadd.ps.512(<16 x float>, <16 x float>, <16 x float>) nounwind readnone

define <8 x double> @test_x86_vfmadd_pd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_x86_vfmadd_pd_z
  ; CHECK: vfmadd213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.vfmadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) nounwind
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.fma.vfmadd.pd.512(<8 x double>, <8 x double>, <8 x double>) nounwind readnone

define <16 x float> @test_x86_vfmsubps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_x86_vfmsubps_z
  ; CHECK: vfmsub213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.vfmsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) nounwind
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.fma.vfmsub.ps.512(<16 x float>, <16 x float>, <16 x float>) nounwind readnone

define <8 x double> @test_x86_vfmsubpd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_x86_vfmsubpd_z
  ; CHECK: vfmsub213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.vfmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) nounwind
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.fma.vfmsub.pd.512(<8 x double>, <8 x double>, <8 x double>) nounwind readnone

define <16 x float> @test_x86_vfnmadd_ps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_x86_vfnmadd_ps_z
  ; CHECK: vfnmadd213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.vfnmadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) nounwind
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.fma.vfnmadd.ps.512(<16 x float>, <16 x float>, <16 x float>) nounwind readnone

define <8 x double> @test_x86_vfnmadd_pd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_x86_vfnmadd_pd_z
  ; CHECK: vfnmadd213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.vfnmadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) nounwind
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.fma.vfnmadd.pd.512(<8 x double>, <8 x double>, <8 x double>) nounwind readnone

define <16 x float> @test_x86_vfnmsubps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_x86_vfnmsubps_z
  ; CHECK: vfnmsub213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.vfnmsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) nounwind
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.fma.vfnmsub.ps.512(<16 x float>, <16 x float>, <16 x float>) nounwind readnone

define <8 x double> @test_x86_vfnmsubpd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_x86_vfnmsubpd_z
  ; CHECK: vfnmsub213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.vfnmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) nounwind
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.fma.vfnmsub.pd.512(<8 x double>, <8 x double>, <8 x double>) nounwind readnone

define <16 x float> @test_x86_vfmaddsubps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_x86_vfmaddsubps_z
  ; CHECK: vfmaddsub213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.vfmaddsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) nounwind
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.fma.vfmaddsub.ps.512(<16 x float>, <16 x float>, <16 x float>) nounwind readnone

define <8 x double> @test_x86_vfmaddsubpd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_x86_vfmaddsubpd_z
  ; CHECK: vfmaddsub213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.vfmaddsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) nounwind
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.fma.vfmaddsub.pd.512(<8 x double>, <8 x double>, <8 x double>) nounwind readnone

define <16 x float> @test_x86_vfmsubaddps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_x86_vfmsubaddps_z
  ; CHECK: vfmsubadd213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.vfmsubadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) nounwind
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.fma.vfmsubadd.ps.512(<16 x float>, <16 x float>, <16 x float>) nounwind readnone

define <8 x double> @test_x86_vfmsubaddpd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_x86_vfmsubaddpd_z
  ; CHECK: vfmsubadd213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.vfmsubadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) nounwind
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.fma.vfmsubadd.pd.512(<8 x double>, <8 x double>, <8 x double>) nounwind readnone

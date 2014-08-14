; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl --show-mc-encoding | FileCheck %s

define <16 x float> @test_x86_vfmadd_ps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_x86_vfmadd_ps_z
  ; CHECK: vfmadd213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.mask.vfmadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 4) nounwind
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.fma.mask.vfmadd.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32) nounwind readnone

define <8 x double> @test_x86_vfmadd_pd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_x86_vfmadd_pd_z
  ; CHECK: vfmadd213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.mask.vfmadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 4) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_fmadd_pd(<8 x double> %a, <8 x double> %b, <8 x double> %c, i8 %mask) {
; CHECK-LABEL: test_mask_fmadd_pd:
; CHECK: vfmadd213pd %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x49,0xa8,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfmadd.pd.512(<8 x double> %a, <8 x double> %b, <8 x double> %c, i8 %mask, i32 4)
  ret <8 x double> %res
}

declare <8 x double> @llvm.x86.fma.mask.vfmadd.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32)

define <16 x float> @test_x86_vfmsubps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_x86_vfmsubps_z
  ; CHECK: vfmsub213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.mask.vfmsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 4) nounwind
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.fma.mask.vfmsub.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32) nounwind readnone

define <8 x double> @test_x86_vfmsubpd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_x86_vfmsubpd_z
  ; CHECK: vfmsub213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.mask.vfmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 4) nounwind
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.fma.mask.vfmsub.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32) nounwind readnone

define <16 x float> @test_x86_vfnmadd_ps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_x86_vfnmadd_ps_z
  ; CHECK: vfnmadd213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.mask.vfnmadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 4) nounwind
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.fma.mask.vfnmadd.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32) nounwind readnone

define <8 x double> @test_x86_vfnmadd_pd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_x86_vfnmadd_pd_z
  ; CHECK: vfnmadd213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.mask.vfnmadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 4) nounwind
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.fma.mask.vfnmadd.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32) nounwind readnone

define <16 x float> @test_x86_vfnmsubps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_x86_vfnmsubps_z
  ; CHECK: vfnmsub213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.mask.vfnmsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 4) nounwind
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.fma.mask.vfnmsub.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32) nounwind readnone

define <8 x double> @test_x86_vfnmsubpd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_x86_vfnmsubpd_z
  ; CHECK: vfnmsub213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.mask.vfnmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 4) nounwind
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.fma.mask.vfnmsub.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32) nounwind readnone

define <16 x float> @test_x86_vfmaddsubps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_x86_vfmaddsubps_z
  ; CHECK: vfmaddsub213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.mask.vfmaddsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 4) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_fmaddsub_ps(<16 x float> %a, <16 x float> %b, <16 x float> %c, i16 %mask) {
; CHECK-LABEL: test_mask_fmaddsub_ps:
; CHECK: vfmaddsub213ps %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x49,0xa6,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmaddsub.ps.512(<16 x float> %a, <16 x float> %b, <16 x float> %c, i16 %mask, i32 4)
  ret <16 x float> %res
}

declare <16 x float> @llvm.x86.fma.mask.vfmaddsub.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32) nounwind readnone

define <8 x double> @test_x86_vfmaddsubpd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_x86_vfmaddsubpd_z
  ; CHECK: vfmaddsub213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.mask.vfmaddsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 4) nounwind
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.fma.mask.vfmaddsub.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32) nounwind readnone

define <16 x float> @test_x86_vfmsubaddps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_x86_vfmsubaddps_z
  ; CHECK: vfmsubadd213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.mask.vfmsubadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 4) nounwind
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.fma.mask.vfmsubadd.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32) nounwind readnone

define <8 x double> @test_x86_vfmsubaddpd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_x86_vfmsubaddpd_z
  ; CHECK: vfmsubadd213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.mask.vfmsubadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 4) nounwind
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.fma.mask.vfmsubadd.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32) nounwind readnone

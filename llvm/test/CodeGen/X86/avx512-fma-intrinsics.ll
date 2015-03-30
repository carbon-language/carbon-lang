; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl --show-mc-encoding | FileCheck %s

declare <16 x float> @llvm.x86.fma.mask.vfmadd.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)
declare <8 x double> @llvm.x86.fma.mask.vfmadd.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32)
declare <16 x float> @llvm.x86.fma.mask.vfmsub.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)

define <8 x double> @test_x86_vfmsubpd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_x86_vfmsubpd_z
  ; CHECK: vfmsub213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.mask.vfmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 4) nounwind
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.fma.mask.vfmsub.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32) nounwind readnone

define <8 x double> @test_mask_vfmsub_pd(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmsub_pd
  ; CHECK: vfmsub213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.mask.vfmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask, i32 4) nounwind
  ret <8 x double> %res
}

define <16 x float> @test_x86_vfnmadd_ps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_x86_vfnmadd_ps_z
  ; CHECK: vfnmadd213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.mask.vfnmadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 4) nounwind
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.fma.mask.vfnmadd.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32) nounwind readnone

define <16 x float> @test_mask_vfnmadd_ps(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask) {
  ; CHECK-LABEL: test_mask_vfnmadd_ps
  ; CHECK: vfnmadd213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.mask.vfnmadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask, i32 4) nounwind
  ret <16 x float> %res
}

define <8 x double> @test_x86_vfnmadd_pd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_x86_vfnmadd_pd_z
  ; CHECK: vfnmadd213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.mask.vfnmadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 4) nounwind
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.fma.mask.vfnmadd.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32) nounwind readnone

define <8 x double> @test_mask_vfnmadd_pd(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmadd_pd
  ; CHECK: vfnmadd213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.mask.vfnmadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask, i32 4) nounwind
  ret <8 x double> %res
}

define <16 x float> @test_x86_vfnmsubps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_x86_vfnmsubps_z
  ; CHECK: vfnmsub213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.mask.vfnmsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 4) nounwind
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.fma.mask.vfnmsub.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32) nounwind readnone

define <16 x float> @test_mask_vfnmsub_ps(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask) {
  ; CHECK-LABEL: test_mask_vfnmsub_ps
  ; CHECK: vfnmsub213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.mask.vfnmsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask, i32 4) nounwind
  ret <16 x float> %res
}

define <8 x double> @test_x86_vfnmsubpd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_x86_vfnmsubpd_z
  ; CHECK: vfnmsub213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.mask.vfnmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 4) nounwind
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.fma.mask.vfnmsub.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32) nounwind readnone

define <8 x double> @test_mask_vfnmsub_pd(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfnmsub_pd
  ; CHECK: vfnmsub213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.mask.vfnmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask, i32 4) nounwind
  ret <8 x double> %res
}

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

define <8 x double> @test_mask_vfmaddsub_pd(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmaddsub_pd
  ; CHECK: vfmaddsub213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.mask.vfmaddsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask, i32 4) nounwind
  ret <8 x double> %res
}

define <16 x float> @test_x86_vfmsubaddps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_x86_vfmsubaddps_z
  ; CHECK: vfmsubadd213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.mask.vfmsubadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 4) nounwind
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.fma.mask.vfmsubadd.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32) nounwind readnone

define <16 x float> @test_mask_vfmsubadd_ps(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask) {
  ; CHECK-LABEL: test_mask_vfmsubadd_ps
  ; CHECK: vfmsubadd213ps %zmm
  %res = call <16 x float> @llvm.x86.fma.mask.vfmsubadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask, i32 4) nounwind
  ret <16 x float> %res
}

define <8 x double> @test_x86_vfmsubaddpd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_x86_vfmsubaddpd_z
  ; CHECK: vfmsubadd213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.mask.vfmsubadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 4) nounwind
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.fma.mask.vfmsubadd.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32) nounwind readnone

define <8 x double> @test_mask_vfmsubadd_pd(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_vfmsubadd_pd
  ; CHECK: vfmsubadd213pd %zmm
  %res = call <8 x double> @llvm.x86.fma.mask.vfmsubadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask, i32 4) nounwind
  ret <8 x double> %res
}

define <16 x float> @test_mask_round_vfmadd512_ps_rrb_rne(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_ps_rrb_rne
  ; CHECK: vfmadd213ps  {rn-sae}, %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x19,0xa8,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask, i32 0) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmadd512_ps_rrb_rtn(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_ps_rrb_rtn
  ; CHECK: vfmadd213ps  {rd-sae}, %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x39,0xa8,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask, i32 1) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmadd512_ps_rrb_rtp(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_ps_rrb_rtp
  ; CHECK: vfmadd213ps  {ru-sae}, %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x59,0xa8,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask, i32 2) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmadd512_ps_rrb_rtz(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_ps_rrb_rtz
  ; CHECK: vfmadd213ps  {rz-sae}, %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x79,0xa8,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask, i32 3) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmadd512_ps_rrb_current(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_ps_rrb_current
  ; CHECK: vfmadd213ps  %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x49,0xa8,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask, i32 4) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmadd512_ps_rrbz_rne(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_ps_rrbz_rne
  ; CHECK: vfmadd213ps  {rn-sae}, %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0x75,0x18,0xa8,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 0) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmadd512_ps_rrbz_rtn(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_ps_rrbz_rtn
  ; CHECK: vfmadd213ps  {rd-sae}, %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0x75,0x38,0xa8,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 1) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmadd512_ps_rrbz_rtp(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_ps_rrbz_rtp
  ; CHECK: vfmadd213ps  {ru-sae}, %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0x75,0x58,0xa8,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 2) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmadd512_ps_rrbz_rtz(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_ps_rrbz_rtz
  ; CHECK: vfmadd213ps  {rz-sae}, %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0x75,0x78,0xa8,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 3) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmadd512_ps_rrbz_current(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_ps_rrbz_current
  ; CHECK: vfmadd213ps  %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0x75,0x48,0xa8,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmadd.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 4) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmsub512_ps_rrb_rne(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask) {
  ; CHECK-LABEL: test_mask_round_vfmsub512_ps_rrb_rne
  ; CHECK: vfmsub213ps  {rn-sae}, %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x19,0xaa,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask, i32 0) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmsub512_ps_rrb_rtn(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask) {
  ; CHECK-LABEL: test_mask_round_vfmsub512_ps_rrb_rtn
  ; CHECK: vfmsub213ps  {rd-sae}, %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x39,0xaa,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask, i32 1) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmsub512_ps_rrb_rtp(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask) {
  ; CHECK-LABEL: test_mask_round_vfmsub512_ps_rrb_rtp
  ; CHECK: vfmsub213ps  {ru-sae}, %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x59,0xaa,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask, i32 2) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmsub512_ps_rrb_rtz(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask) {
  ; CHECK-LABEL: test_mask_round_vfmsub512_ps_rrb_rtz
  ; CHECK: vfmsub213ps  {rz-sae}, %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x79,0xaa,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask, i32 3) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmsub512_ps_rrb_current(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask) {
  ; CHECK-LABEL: test_mask_round_vfmsub512_ps_rrb_current
  ; CHECK: vfmsub213ps  %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0x75,0x49,0xaa,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 %mask, i32 4) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmsub512_ps_rrbz_rne(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_mask_round_vfmsub512_ps_rrbz_rne
  ; CHECK: vfmsub213ps  {rn-sae}, %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0x75,0x18,0xaa,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 0) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmsub512_ps_rrbz_rtn(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_mask_round_vfmsub512_ps_rrbz_rtn
  ; CHECK: vfmsub213ps  {rd-sae}, %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0x75,0x38,0xaa,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 1) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmsub512_ps_rrbz_rtp(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_mask_round_vfmsub512_ps_rrbz_rtp
  ; CHECK: vfmsub213ps  {ru-sae}, %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0x75,0x58,0xaa,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 2) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmsub512_ps_rrbz_rtz(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_mask_round_vfmsub512_ps_rrbz_rtz
  ; CHECK: vfmsub213ps  {rz-sae}, %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0x75,0x78,0xaa,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 3) nounwind
  ret <16 x float> %res
}

define <16 x float> @test_mask_round_vfmsub512_ps_rrbz_current(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK-LABEL: test_mask_round_vfmsub512_ps_rrbz_current
  ; CHECK: vfmsub213ps  %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0x75,0x48,0xaa,0xc2]
  %res = call <16 x float> @llvm.x86.fma.mask.vfmsub.ps.512(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2, i16 -1, i32 4) nounwind
  ret <16 x float> %res
}

define <8 x double> @test_mask_round_vfmadd512_pd_rrb_rne(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_pd_rrb_rne
  ; CHECK: vfmadd213pd  {rn-sae}, %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x19,0xa8,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfmadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask, i32 0) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfmadd512_pd_rrb_rtn(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_pd_rrb_rtn
  ; CHECK: vfmadd213pd  {rd-sae}, %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x39,0xa8,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfmadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask, i32 1) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfmadd512_pd_rrb_rtp(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_pd_rrb_rtp
  ; CHECK: vfmadd213pd  {ru-sae}, %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x59,0xa8,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfmadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask, i32 2) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfmadd512_pd_rrb_rtz(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_pd_rrb_rtz
  ; CHECK: vfmadd213pd  {rz-sae}, %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x79,0xa8,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfmadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask, i32 3) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfmadd512_pd_rrb_current(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_pd_rrb_current
  ; CHECK: vfmadd213pd  %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x49,0xa8,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfmadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask, i32 4) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfmadd512_pd_rrbz_rne(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_pd_rrbz_rne
  ; CHECK: vfmadd213pd  {rn-sae}, %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0xf5,0x18,0xa8,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfmadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 0) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfmadd512_pd_rrbz_rtn(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_pd_rrbz_rtn
  ; CHECK: vfmadd213pd  {rd-sae}, %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0xf5,0x38,0xa8,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfmadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 1) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfmadd512_pd_rrbz_rtp(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_pd_rrbz_rtp
  ; CHECK: vfmadd213pd  {ru-sae}, %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0xf5,0x58,0xa8,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfmadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 2) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfmadd512_pd_rrbz_rtz(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_pd_rrbz_rtz
  ; CHECK: vfmadd213pd  {rz-sae}, %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0xf5,0x78,0xa8,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfmadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 3) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfmadd512_pd_rrbz_current(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_mask_round_vfmadd512_pd_rrbz_current
  ; CHECK: vfmadd213pd  %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0xf5,0x48,0xa8,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfmadd.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 4) nounwind
  ret <8 x double> %res
}


define <8 x double> @test_mask_round_vfnmsub512_pd_rrb_rne(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_round_vfnmsub512_pd_rrb_rne
  ; CHECK: vfnmsub213pd  {rn-sae}, %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x19,0xae,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfnmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask, i32 0) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfnmsub512_pd_rrb_rtn(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_round_vfnmsub512_pd_rrb_rtn
  ; CHECK: vfnmsub213pd  {rd-sae}, %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x39,0xae,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfnmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask, i32 1) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfnmsub512_pd_rrb_rtp(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_round_vfnmsub512_pd_rrb_rtp
  ; CHECK: vfnmsub213pd  {ru-sae}, %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x59,0xae,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfnmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask, i32 2) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfnmsub512_pd_rrb_rtz(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_round_vfnmsub512_pd_rrb_rtz
  ; CHECK: vfnmsub213pd  {rz-sae}, %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x79,0xae,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfnmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask, i32 3) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfnmsub512_pd_rrb_current(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask) {
  ; CHECK-LABEL: test_mask_round_vfnmsub512_pd_rrb_current
  ; CHECK: vfnmsub213pd  %zmm2, %zmm1, %zmm0 {%k1} ## encoding: [0x62,0xf2,0xf5,0x49,0xae,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfnmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 %mask, i32 4) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfnmsub512_pd_rrbz_rne(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_mask_round_vfnmsub512_pd_rrbz_rne
  ; CHECK: vfnmsub213pd  {rn-sae}, %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0xf5,0x18,0xae,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfnmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 0) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfnmsub512_pd_rrbz_rtn(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_mask_round_vfnmsub512_pd_rrbz_rtn
  ; CHECK: vfnmsub213pd  {rd-sae}, %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0xf5,0x38,0xae,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfnmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 1) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfnmsub512_pd_rrbz_rtp(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_mask_round_vfnmsub512_pd_rrbz_rtp
  ; CHECK: vfnmsub213pd  {ru-sae}, %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0xf5,0x58,0xae,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfnmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 2) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfnmsub512_pd_rrbz_rtz(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_mask_round_vfnmsub512_pd_rrbz_rtz
  ; CHECK: vfnmsub213pd  {rz-sae}, %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0xf5,0x78,0xae,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfnmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 3) nounwind
  ret <8 x double> %res
}

define <8 x double> @test_mask_round_vfnmsub512_pd_rrbz_current(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK-LABEL: test_mask_round_vfnmsub512_pd_rrbz_current
  ; CHECK: vfnmsub213pd  %zmm2, %zmm1, %zmm0 ## encoding: [0x62,0xf2,0xf5,0x48,0xae,0xc2]
  %res = call <8 x double> @llvm.x86.fma.mask.vfnmsub.pd.512(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2, i8 -1, i32 4) nounwind
  ret <8 x double> %res
}

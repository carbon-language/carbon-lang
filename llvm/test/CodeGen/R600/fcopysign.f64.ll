; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare double @llvm.copysign.f64(double, double) nounwind readnone
declare <2 x double> @llvm.copysign.v2f64(<2 x double>, <2 x double>) nounwind readnone
declare <4 x double> @llvm.copysign.v4f64(<4 x double>, <4 x double>) nounwind readnone

; FUNC-LABEL: {{^}}test_copysign_f64:
; SI-DAG: s_load_dwordx2 s{{\[}}[[SMAG_LO:[0-9]+]]:[[SMAG_HI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dwordx2 s{{\[}}[[SSIGN_LO:[0-9]+]]:[[SSIGN_HI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0xd
; SI-DAG: v_mov_b32_e32 v[[VSIGN_HI:[0-9]+]], s[[SSIGN_HI]]
; SI-DAG: v_mov_b32_e32 v[[VMAG_HI:[0-9]+]], s[[SMAG_HI]]
; SI-DAG: s_mov_b32 [[SCONST:s[0-9]+]], 0x7fffffff
; SI: v_bfi_b32 v[[VRESULT_HI:[0-9]+]], [[SCONST]], v[[VMAG_HI]], v[[VSIGN_HI]]
; SI: v_mov_b32_e32 v[[VMAG_LO:[0-9]+]], s[[SMAG_LO]]
; SI: buffer_store_dwordx2 v{{\[}}[[VMAG_LO]]:[[VRESULT_HI]]{{\]}}
; SI: s_endpgm
define void @test_copysign_f64(double addrspace(1)* %out, double %mag, double %sign) nounwind {
  %result = call double @llvm.copysign.f64(double %mag, double %sign)
  store double %result, double addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}test_copysign_v2f64:
; SI: s_endpgm
define void @test_copysign_v2f64(<2 x double> addrspace(1)* %out, <2 x double> %mag, <2 x double> %sign) nounwind {
  %result = call <2 x double> @llvm.copysign.v2f64(<2 x double> %mag, <2 x double> %sign)
  store <2 x double> %result, <2 x double> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}test_copysign_v4f64:
; SI: s_endpgm
define void @test_copysign_v4f64(<4 x double> addrspace(1)* %out, <4 x double> %mag, <4 x double> %sign) nounwind {
  %result = call <4 x double> @llvm.copysign.v4f64(<4 x double> %mag, <4 x double> %sign)
  store <4 x double> %result, <4 x double> addrspace(1)* %out, align 8
  ret void
}

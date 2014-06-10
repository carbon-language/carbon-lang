; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s


declare float @llvm.copysign.f32(float, float) nounwind readnone
declare <2 x float> @llvm.copysign.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.copysign.v4f32(<4 x float>, <4 x float>) nounwind readnone

; Try to identify arg based on higher address.
; FUNC-LABEL: @test_copysign_f32:
; SI: S_LOAD_DWORD [[SSIGN:s[0-9]+]], {{.*}} 0xc
; SI: V_MOV_B32_e32 [[VSIGN:v[0-9]+]], [[SSIGN]]
; SI-DAG: S_LOAD_DWORD [[SMAG:s[0-9]+]], {{.*}} 0xb
; SI-DAG: V_MOV_B32_e32 [[VMAG:v[0-9]+]], [[SMAG]]
; SI-DAG: S_MOV_B32 [[SCONST:s[0-9]+]], 0x7fffffff
; SI: V_BFI_B32 [[RESULT:v[0-9]+]], [[SCONST]], [[VMAG]], [[VSIGN]]
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM

; EG: BFI_INT
define void @test_copysign_f32(float addrspace(1)* %out, float %mag, float %sign) nounwind {
  %result = call float @llvm.copysign.f32(float %mag, float %sign)
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_copysign_v2f32:
; SI: S_ENDPGM

; EG: BFI_INT
; EG: BFI_INT
define void @test_copysign_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %mag, <2 x float> %sign) nounwind {
  %result = call <2 x float> @llvm.copysign.v2f32(<2 x float> %mag, <2 x float> %sign)
  store <2 x float> %result, <2 x float> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @test_copysign_v4f32:
; SI: S_ENDPGM

; EG: BFI_INT
; EG: BFI_INT
; EG: BFI_INT
; EG: BFI_INT
define void @test_copysign_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %mag, <4 x float> %sign) nounwind {
  %result = call <4 x float> @llvm.copysign.v4f32(<4 x float> %mag, <4 x float> %sign)
  store <4 x float> %result, <4 x float> addrspace(1)* %out, align 16
  ret void
}


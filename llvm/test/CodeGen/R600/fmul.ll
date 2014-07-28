; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s


; FUNC-LABEL: @fmul_f32
; R600: MUL_IEEE {{\** *}}{{T[0-9]+\.[XYZW]}}, KC0[2].Z, KC0[2].W

; SI: V_MUL_F32
define void @fmul_f32(float addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fmul float %a, %b
  store float %0, float addrspace(1)* %out
  ret void
}

declare float @llvm.R600.load.input(i32) readnone

declare void @llvm.AMDGPU.store.output(float, i32)

; FUNC-LABEL: @fmul_v2f32
; R600: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}
; R600: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}

; SI: V_MUL_F32
; SI: V_MUL_F32
define void @fmul_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b) {
entry:
  %0 = fmul <2 x float> %a, %b
  store <2 x float> %0, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @fmul_v4f32
; R600: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; SI: V_MUL_F32
; SI: V_MUL_F32
; SI: V_MUL_F32
; SI: V_MUL_F32
define void @fmul_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x float> addrspace(1)* %in, i32 1
  %a = load <4 x float> addrspace(1) * %in
  %b = load <4 x float> addrspace(1) * %b_ptr
  %result = fmul <4 x float> %a, %b
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}

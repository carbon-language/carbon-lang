; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare float @llvm.fma.f32(float, float, float) nounwind readnone
declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

declare double @llvm.fma.f64(double, double, double) nounwind readnone
declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>) nounwind readnone
declare <4 x double> @llvm.fma.v4f64(<4 x double>, <4 x double>, <4 x double>) nounwind readnone

; FUNC-LABEL: @fma_f32
; SI: V_FMA_F32 {{v[0-9]+, v[0-9]+, v[0-9]+, v[0-9]+}}
define void @fma_f32(float addrspace(1)* %out, float addrspace(1)* %in1,
                     float addrspace(1)* %in2, float addrspace(1)* %in3) {
   %r0 = load float addrspace(1)* %in1
   %r1 = load float addrspace(1)* %in2
   %r2 = load float addrspace(1)* %in3
   %r3 = tail call float @llvm.fma.f32(float %r0, float %r1, float %r2)
   store float %r3, float addrspace(1)* %out
   ret void
}

; FUNC-LABEL: @fma_v2f32
; SI: V_FMA_F32
; SI: V_FMA_F32
define void @fma_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %in1,
                       <2 x float> addrspace(1)* %in2, <2 x float> addrspace(1)* %in3) {
   %r0 = load <2 x float> addrspace(1)* %in1
   %r1 = load <2 x float> addrspace(1)* %in2
   %r2 = load <2 x float> addrspace(1)* %in3
   %r3 = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %r0, <2 x float> %r1, <2 x float> %r2)
   store <2 x float> %r3, <2 x float> addrspace(1)* %out
   ret void
}

; FUNC-LABEL: @fma_v4f32
; SI: V_FMA_F32
; SI: V_FMA_F32
; SI: V_FMA_F32
; SI: V_FMA_F32
define void @fma_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in1,
                       <4 x float> addrspace(1)* %in2, <4 x float> addrspace(1)* %in3) {
   %r0 = load <4 x float> addrspace(1)* %in1
   %r1 = load <4 x float> addrspace(1)* %in2
   %r2 = load <4 x float> addrspace(1)* %in3
   %r3 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %r0, <4 x float> %r1, <4 x float> %r2)
   store <4 x float> %r3, <4 x float> addrspace(1)* %out
   ret void
}

; FUNC-LABEL: @fma_f64
; SI: V_FMA_F64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define void @fma_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2, double addrspace(1)* %in3) {
   %r0 = load double addrspace(1)* %in1
   %r1 = load double addrspace(1)* %in2
   %r2 = load double addrspace(1)* %in3
   %r3 = tail call double @llvm.fma.f64(double %r0, double %r1, double %r2)
   store double %r3, double addrspace(1)* %out
   ret void
}

; FUNC-LABEL: @fma_v2f64
; SI: V_FMA_F64
; SI: V_FMA_F64
define void @fma_v2f64(<2 x double> addrspace(1)* %out, <2 x double> addrspace(1)* %in1,
                       <2 x double> addrspace(1)* %in2, <2 x double> addrspace(1)* %in3) {
   %r0 = load <2 x double> addrspace(1)* %in1
   %r1 = load <2 x double> addrspace(1)* %in2
   %r2 = load <2 x double> addrspace(1)* %in3
   %r3 = tail call <2 x double> @llvm.fma.v2f64(<2 x double> %r0, <2 x double> %r1, <2 x double> %r2)
   store <2 x double> %r3, <2 x double> addrspace(1)* %out
   ret void
}

; FUNC-LABEL: @fma_v4f64
; SI: V_FMA_F64
; SI: V_FMA_F64
; SI: V_FMA_F64
; SI: V_FMA_F64
define void @fma_v4f64(<4 x double> addrspace(1)* %out, <4 x double> addrspace(1)* %in1,
                       <4 x double> addrspace(1)* %in2, <4 x double> addrspace(1)* %in3) {
   %r0 = load <4 x double> addrspace(1)* %in1
   %r1 = load <4 x double> addrspace(1)* %in2
   %r2 = load <4 x double> addrspace(1)* %in3
   %r3 = tail call <4 x double> @llvm.fma.v4f64(<4 x double> %r0, <4 x double> %r1, <4 x double> %r2)
   store <4 x double> %r3, <4 x double> addrspace(1)* %out
   ret void
}

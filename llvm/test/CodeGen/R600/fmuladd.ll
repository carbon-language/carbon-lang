; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s

; CHECK: @fmuladd_f32
; CHECK: V_MAD_F32 {{v[0-9]+, v[0-9]+, v[0-9]+, v[0-9]+}}

define void @fmuladd_f32(float addrspace(1)* %out, float addrspace(1)* %in1,
                         float addrspace(1)* %in2, float addrspace(1)* %in3) {
   %r0 = load float addrspace(1)* %in1
   %r1 = load float addrspace(1)* %in2
   %r2 = load float addrspace(1)* %in3
   %r3 = tail call float @llvm.fmuladd.f32(float %r0, float %r1, float %r2)
   store float %r3, float addrspace(1)* %out
   ret void
}

declare float @llvm.fmuladd.f32(float, float, float)

; CHECK: @fmuladd_f64
; CHECK: V_FMA_F64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}

define void @fmuladd_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                         double addrspace(1)* %in2, double addrspace(1)* %in3) {
   %r0 = load double addrspace(1)* %in1
   %r1 = load double addrspace(1)* %in2
   %r2 = load double addrspace(1)* %in3
   %r3 = tail call double @llvm.fmuladd.f64(double %r0, double %r1, double %r2)
   store double %r3, double addrspace(1)* %out
   ret void
}

declare double @llvm.fmuladd.f64(double, double, double)

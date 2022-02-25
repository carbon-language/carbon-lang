; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=gfx90a -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GFX90A -check-prefix=FUNC %s

declare double @llvm.fma.f64(double, double, double) nounwind readnone
declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>) nounwind readnone
declare <4 x double> @llvm.fma.v4f64(<4 x double>, <4 x double>, <4 x double>) nounwind readnone
declare double @llvm.fabs.f64(double) nounwind readnone

; FUNC-LABEL: {{^}}fma_f64:
; SI: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; GFX90A: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fma_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2, double addrspace(1)* %in3) {
   %r0 = load double, double addrspace(1)* %in1
   %r1 = load double, double addrspace(1)* %in2
   %r2 = load double, double addrspace(1)* %in3
   %r3 = tail call double @llvm.fma.f64(double %r0, double %r1, double %r2)
   store double %r3, double addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}fma_v2f64:
; SI: v_fma_f64
; SI: v_fma_f64
; GFX90A: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; GFX90A: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fma_v2f64(<2 x double> addrspace(1)* %out, <2 x double> addrspace(1)* %in1,
                       <2 x double> addrspace(1)* %in2, <2 x double> addrspace(1)* %in3) {
   %r0 = load <2 x double>, <2 x double> addrspace(1)* %in1
   %r1 = load <2 x double>, <2 x double> addrspace(1)* %in2
   %r2 = load <2 x double>, <2 x double> addrspace(1)* %in3
   %r3 = tail call <2 x double> @llvm.fma.v2f64(<2 x double> %r0, <2 x double> %r1, <2 x double> %r2)
   store <2 x double> %r3, <2 x double> addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}fma_v4f64:
; SI: v_fma_f64
; SI: v_fma_f64
; SI: v_fma_f64
; SI: v_fma_f64
; GFX90A: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; GFX90A: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; GFX90A: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; GFX90A: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fma_v4f64(<4 x double> addrspace(1)* %out, <4 x double> addrspace(1)* %in1,
                       <4 x double> addrspace(1)* %in2, <4 x double> addrspace(1)* %in3) {
   %r0 = load <4 x double>, <4 x double> addrspace(1)* %in1
   %r1 = load <4 x double>, <4 x double> addrspace(1)* %in2
   %r2 = load <4 x double>, <4 x double> addrspace(1)* %in3
   %r3 = tail call <4 x double> @llvm.fma.v4f64(<4 x double> %r0, <4 x double> %r1, <4 x double> %r2)
   store <4 x double> %r3, <4 x double> addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}fma_f64_abs_src0:
; SI: v_fma_f64 {{v\[[0-9]+:[0-9]+\], |v\[[0-9]+:[0-9]+\]|, v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; GFX90A: v_fmac_f64_e64 {{v\[[0-9]+:[0-9]+\], |v\[[0-9]+:[0-9]+\]|, v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fma_f64_abs_src0(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2, double addrspace(1)* %in3) {
   %r0 = load double, double addrspace(1)* %in1
   %r1 = load double, double addrspace(1)* %in2
   %r2 = load double, double addrspace(1)* %in3
   %fabs = call double @llvm.fabs.f64(double %r0)
   %r3 = tail call double @llvm.fma.f64(double %fabs, double %r1, double %r2)
   store double %r3, double addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}fma_f64_abs_src1:
; SI: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], \|v\[[0-9]+:[0-9]+\]\|, v\[[0-9]+:[0-9]+\]}}
; GFX90A: v_fmac_f64_e64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], \|v\[[0-9]+:[0-9]+\]\|}}
define amdgpu_kernel void @fma_f64_abs_src1(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2, double addrspace(1)* %in3) {
   %r0 = load double, double addrspace(1)* %in1
   %r1 = load double, double addrspace(1)* %in2
   %r2 = load double, double addrspace(1)* %in3
   %fabs = call double @llvm.fabs.f64(double %r1)
   %r3 = tail call double @llvm.fma.f64(double %r0, double %fabs, double %r2)
   store double %r3, double addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}fma_f64_abs_src2:
; SI: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], \|v\[[0-9]+:[0-9]+\]\|}}
; GFX90A: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], \|v\[[0-9]+:[0-9]+\]\|}}
define amdgpu_kernel void @fma_f64_abs_src2(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2, double addrspace(1)* %in3) {
   %r0 = load double, double addrspace(1)* %in1
   %r1 = load double, double addrspace(1)* %in2
   %r2 = load double, double addrspace(1)* %in3
   %fabs = call double @llvm.fabs.f64(double %r2)
   %r3 = tail call double @llvm.fma.f64(double %r0, double %r1, double %fabs)
   store double %r3, double addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}fma_f64_neg_src0:
; SI: v_fma_f64 {{v\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; GFX90A: v_fmac_f64_e64 {{v\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fma_f64_neg_src0(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2, double addrspace(1)* %in3) {
   %r0 = load double, double addrspace(1)* %in1
   %r1 = load double, double addrspace(1)* %in2
   %r2 = load double, double addrspace(1)* %in3
   %fsub = fsub double -0.000000e+00, %r0
   %r3 = tail call double @llvm.fma.f64(double %fsub, double %r1, double %r2)
   store double %r3, double addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}fma_f64_neg_src1:
; SI: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; GFX90A: v_fmac_f64_e64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fma_f64_neg_src1(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2, double addrspace(1)* %in3) {
   %r0 = load double, double addrspace(1)* %in1
   %r1 = load double, double addrspace(1)* %in2
   %r2 = load double, double addrspace(1)* %in3
   %fsub = fsub double -0.000000e+00, %r1
   %r3 = tail call double @llvm.fma.f64(double %r0, double %fsub, double %r2)
   store double %r3, double addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}fma_f64_neg_src2:
; SI: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
; GFX90A: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fma_f64_neg_src2(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2, double addrspace(1)* %in3) {
   %r0 = load double, double addrspace(1)* %in1
   %r1 = load double, double addrspace(1)* %in2
   %r2 = load double, double addrspace(1)* %in3
   %fsub = fsub double -0.000000e+00, %r2
   %r3 = tail call double @llvm.fma.f64(double %r0, double %r1, double %fsub)
   store double %r3, double addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}fma_f64_abs_neg_src0:
; SI: v_fma_f64 {{v\[[0-9]+:[0-9]+\], -\|v\[[0-9]+:[0-9]+\]\|, v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; GFX90A: v_fmac_f64_e64 {{v\[[0-9]+:[0-9]+\], -\|v\[[0-9]+:[0-9]+\]\|, v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fma_f64_abs_neg_src0(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2, double addrspace(1)* %in3) {
   %r0 = load double, double addrspace(1)* %in1
   %r1 = load double, double addrspace(1)* %in2
   %r2 = load double, double addrspace(1)* %in3
   %fabs = call double @llvm.fabs.f64(double %r0)
   %fsub = fsub double -0.000000e+00, %fabs
   %r3 = tail call double @llvm.fma.f64(double %fsub, double %r1, double %r2)
   store double %r3, double addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}fma_f64_abs_neg_src1:
; SI: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -\|v\[[0-9]+:[0-9]+\]\|, v\[[0-9]+:[0-9]+\]}}
; GFX90A: v_fmac_f64_e64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -\|v\[[0-9]+:[0-9]+\]\|}}
define amdgpu_kernel void @fma_f64_abs_neg_src1(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2, double addrspace(1)* %in3) {
   %r0 = load double, double addrspace(1)* %in1
   %r1 = load double, double addrspace(1)* %in2
   %r2 = load double, double addrspace(1)* %in3
   %fabs = call double @llvm.fabs.f64(double %r1)
   %fsub = fsub double -0.000000e+00, %fabs
   %r3 = tail call double @llvm.fma.f64(double %r0, double %fsub, double %r2)
   store double %r3, double addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}fma_f64_abs_neg_src2:
; SI: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -\|v\[[0-9]+:[0-9]+\]\|}}
; GFX90A: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -\|v\[[0-9]+:[0-9]+\]\|}}
define amdgpu_kernel void @fma_f64_abs_neg_src2(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2, double addrspace(1)* %in3) {
   %r0 = load double, double addrspace(1)* %in1
   %r1 = load double, double addrspace(1)* %in2
   %r2 = load double, double addrspace(1)* %in3
   %fabs = call double @llvm.fabs.f64(double %r2)
   %fsub = fsub double -0.000000e+00, %fabs
   %r3 = tail call double @llvm.fma.f64(double %r0, double %r1, double %fsub)
   store double %r3, double addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}fma_f64_lit_src0:
; SI: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], 2.0, v\[[0-9]+:[0-9]+\]}}
; GFX90A: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], 2.0, v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fma_f64_lit_src0(double addrspace(1)* %out,
                     double addrspace(1)* %in2, double addrspace(1)* %in3) {
   %r1 = load double, double addrspace(1)* %in2
   %r2 = load double, double addrspace(1)* %in3
   %r3 = tail call double @llvm.fma.f64(double +2.0, double %r1, double %r2)
   store double %r3, double addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}fma_f64_lit_src1:
; SI: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], 2.0, v\[[0-9]+:[0-9]+\]}}
; GFX90A: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], 2.0, v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fma_f64_lit_src1(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in3) {
   %r0 = load double, double addrspace(1)* %in1
   %r2 = load double, double addrspace(1)* %in3
   %r3 = tail call double @llvm.fma.f64(double %r0, double +2.0, double %r2)
   store double %r3, double addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}fma_f64_lit_src2:
; SI: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], 2.0}}
; GFX90A: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], 2.0}}
define amdgpu_kernel void @fma_f64_lit_src2(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2) {
   %r0 = load double, double addrspace(1)* %in1
   %r1 = load double, double addrspace(1)* %in2
   %r3 = tail call double @llvm.fma.f64(double %r0, double %r1, double +2.0)
   store double %r3, double addrspace(1)* %out
   ret void
}

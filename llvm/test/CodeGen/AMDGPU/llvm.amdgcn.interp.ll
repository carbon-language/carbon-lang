;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=GCN %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefixes=GCN,VI %s

;GCN-LABEL: {{^}}v_interp:
;GCN-NOT: s_wqm
;GCN: s_mov_b32 m0, s{{[0-9]+}}
;GCN: v_interp_p1_f32
;GCN: v_interp_p2_f32
define amdgpu_ps void @v_interp(<16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, i32 inreg, <2 x float>) {
main_body:
  %i = extractelement <2 x float> %4, i32 0
  %j = extractelement <2 x float> %4, i32 1
  %p0_0 = call float @llvm.amdgcn.interp.p1(float %i, i32 0, i32 0, i32 %3)
  %p1_0 = call float @llvm.amdgcn.interp.p2(float %p0_0, float %j, i32 0, i32 0, i32 %3)
  %p0_1 = call float @llvm.amdgcn.interp.p1(float %i, i32 1, i32 0, i32 %3)
  %p1_1 = call float @llvm.amdgcn.interp.p2(float %p0_1, float %j, i32 1, i32 0, i32 %3)
  %const = call float @llvm.amdgcn.interp.mov(i32 2, i32 0, i32 0, i32 %3)
  %w = fadd float %p1_1, %const
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %p0_0, float %p0_0, float %p1_1, float %w)
  ret void
}

; SI won't merge ds memory operations, because of the signed offset bug, so
; we only have check lines for VI.
; VI-LABEL: v_interp_readnone:
; VI: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; VI: ds_write2_b32 v{{[0-9]+}}, [[ZERO]], [[ZERO]] offset1:4
define amdgpu_ps void @v_interp_readnone(float addrspace(3)* %lds) {
  store float 0.0, float addrspace(3)* %lds
  %tmp1 = call float @llvm.amdgcn.interp.mov(i32 2, i32 0, i32 0, i32 0)
  %tmp2 = getelementptr float, float addrspace(3)* %lds, i32 4
  store float 0.0, float addrspace(3)* %tmp2
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %tmp1, float %tmp1, float %tmp1, float %tmp1)
  ret void
}

; Function Attrs: nounwind readnone
declare float @llvm.amdgcn.interp.p1(float, i32, i32, i32) #0

; Function Attrs: nounwind readnone
declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32) #0

declare float @llvm.amdgcn.interp.mov(i32, i32, i32, i32) #0

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { nounwind readnone }

; Make sure we still form mad even when unsafe math or fp-contract is allowed instead of fma.

; RUN: llc -march=amdgcn -mcpu=tahiti -denormal-fp-math-f32=preserve-sign -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=SI -check-prefix=SI-STD  -check-prefix=SI-STD-SAFE -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tahiti -denormal-fp-math-f32=preserve-sign -verify-machineinstrs -fp-contract=fast < %s | FileCheck -enable-var-scope -check-prefix=SI -check-prefix=SI-STD -check-prefix=SI-STD-SAFE -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tahiti -denormal-fp-math-f32=preserve-sign -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -enable-var-scope -check-prefix=SI -check-prefix=SI-STD -check-prefix=SI-STD-UNSAFE -check-prefix=FUNC %s

; FIXME: Remove enable-unsafe-fp-math in RUN line and add flags to IR instrs

; Make sure we don't form mad with denormals
; RUN: llc -march=amdgcn -mcpu=tahiti -denormal-fp-math-f32=ieee -fp-contract=fast -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=SI -check-prefix=SI-DENORM -check-prefix=SI-DENORM-FASTFMAF -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=verde -denormal-fp-math-f32=ieee -fp-contract=fast -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=SI -check-prefix=SI-DENORM -check-prefix=SI-DENORM-SLOWFMAF -check-prefix=FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare float @llvm.fabs.f32(float) #0
declare float @llvm.fma.f32(float, float, float) #0
declare float @llvm.fmuladd.f32(float, float, float) #0

; (fadd (fmul x, y), z) -> (fma x, y, z)
; FUNC-LABEL: {{^}}combine_to_mad_f32_0:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4 glc{{$}}
; SI-DAG: buffer_load_dword [[C:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}

; SI-STD: v_mac_f32_e32 [[C]], [[A]], [[B]]

; SI-DENORM-FASTFMAF: v_fma_f32 [[RESULT:v[0-9]+]], [[A]], [[B]], [[C]]

; SI-DENORM-SLOWFMAF-NOT: v_fma
; SI-DENORM-SLOWFMAF-NOT: v_mad

; SI-DENORM-SLOWFMAF: v_mul_f32_e32 [[TMP:v[0-9]+]], [[A]], [[B]]
; SI-DENORM-SLOWFMAF: v_add_f32_e32 [[RESULT:v[0-9]+]],  [[TMP]], [[C]]

; SI-DENORM: buffer_store_dword [[RESULT]]
; SI-STD: buffer_store_dword [[C]]
define amdgpu_kernel void @combine_to_mad_f32_0(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr float, float addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load volatile float, float addrspace(1)* %gep.0
  %b = load volatile float, float addrspace(1)* %gep.1
  %c = load volatile float, float addrspace(1)* %gep.2

  %mul = fmul float %a, %b
  %fma = fadd float %mul, %c
  store float %fma, float addrspace(1)* %gep.out
  ret void
}

; (fadd (fmul x, y), z) -> (fma x, y, z)
; FUNC-LABEL: {{^}}combine_to_mad_f32_0_2use:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4 glc{{$}}
; SI-DAG: buffer_load_dword [[C:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dword [[D:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:12 glc{{$}}

; SI-STD-DAG: v_mac_f32_e32 [[C]], [[A]], [[B]]
; SI-STD-DAG: v_mac_f32_e32 [[D]], [[A]], [[B]]

; SI-DENORM-FASTFMAF-DAG: v_fma_f32 [[RESULT0:v[0-9]+]], [[A]], [[B]], [[C]]
; SI-DENORM-FASTFMAF-DAG: v_fma_f32 [[RESULT1:v[0-9]+]], [[A]], [[B]], [[D]]

; SI-DENORM-SLOWFMAF: v_mul_f32_e32 [[TMP:v[0-9]+]], [[A]], [[B]]
; SI-DENORM-SLOWFMAF-DAG: v_add_f32_e32 [[RESULT0:v[0-9]+]], [[TMP]], [[C]]
; SI-DENORM-SLOWFMAF-DAG: v_add_f32_e32 [[RESULT1:v[0-9]+]], [[TMP]], [[D]]

; SI-DENORM-DAG: buffer_store_dword [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DENORM-DAG: buffer_store_dword [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4{{$}}
; SI-STD-DAG: buffer_store_dword [[C]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-STD-DAG: buffer_store_dword [[D]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4{{$}}
; SI: s_endpgm
define amdgpu_kernel void @combine_to_mad_f32_0_2use(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr float, float addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr float, float addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr float, float addrspace(1)* %gep.out.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0
  %b = load volatile float, float addrspace(1)* %gep.1
  %c = load volatile float, float addrspace(1)* %gep.2
  %d = load volatile float, float addrspace(1)* %gep.3

  %mul = fmul float %a, %b
  %fma0 = fadd float %mul, %c
  %fma1 = fadd float %mul, %d

  store volatile float %fma0, float addrspace(1)* %gep.out.0
  store volatile float %fma1, float addrspace(1)* %gep.out.1
  ret void
}

; (fadd x, (fmul y, z)) -> (fma y, z, x)
; FUNC-LABEL: {{^}}combine_to_mad_f32_1:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4 glc{{$}}
; SI-DAG: buffer_load_dword [[C:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}

; SI-STD: v_mac_f32_e32 [[C]], [[A]], [[B]]
; SI-DENORM-FASTFMAF: v_fma_f32 [[RESULT:v[0-9]+]], [[A]], [[B]], [[C]]

; SI-DENORM-SLOWFMAF: v_mul_f32_e32 [[TMP:v[0-9]+]], [[A]], [[B]]
; SI-DENORM-SLOWFMAF: v_add_f32_e32 [[RESULT:v[0-9]+]], [[C]], [[TMP]]

; SI-DENORM: buffer_store_dword [[RESULT]]
; SI-STD: buffer_store_dword [[C]]
define amdgpu_kernel void @combine_to_mad_f32_1(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr float, float addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load volatile float, float addrspace(1)* %gep.0
  %b = load volatile float, float addrspace(1)* %gep.1
  %c = load volatile float, float addrspace(1)* %gep.2

  %mul = fmul float %a, %b
  %fma = fadd float %c, %mul
  store float %fma, float addrspace(1)* %gep.out
  ret void
}

; (fsub (fmul x, y), z) -> (fma x, y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_mad_fsub_0_f32:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4 glc{{$}}
; SI-DAG: buffer_load_dword [[C:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}

; SI-STD: v_mad_f32 [[RESULT:v[0-9]+]], [[A]], [[B]], -[[C]]
; SI-DENORM-FASTFMAF: v_fma_f32 [[RESULT:v[0-9]+]], [[A]], [[B]], -[[C]]

; SI-DENORM-SLOWFMAF: v_mul_f32_e32 [[TMP:v[0-9]+]], [[A]], [[B]]
; SI-DENORM-SLOWFMAF: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[TMP]], [[C]]

; SI: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @combine_to_mad_fsub_0_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr float, float addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load volatile float, float addrspace(1)* %gep.0
  %b = load volatile float, float addrspace(1)* %gep.1
  %c = load volatile float, float addrspace(1)* %gep.2

  %mul = fmul float %a, %b
  %fma = fsub float %mul, %c
  store float %fma, float addrspace(1)* %gep.out
  ret void
}

; (fsub (fmul x, y), z) -> (fma x, y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_mad_fsub_0_f32_2use:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4 glc{{$}}
; SI-DAG: buffer_load_dword [[C:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dword [[D:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:12 glc{{$}}

; SI-STD-DAG: v_mad_f32 [[RESULT0:v[0-9]+]], [[A]], [[B]], -[[C]]
; SI-STD-DAG: v_mad_f32 [[RESULT1:v[0-9]+]], [[A]], [[B]], -[[D]]

; SI-DENORM-FASTFMAF-DAG: v_fma_f32 [[RESULT0:v[0-9]+]], [[A]], [[B]], -[[C]]
; SI-DENORM-FASTFMAF-DAG: v_fma_f32 [[RESULT1:v[0-9]+]], [[A]], [[B]], -[[D]]

; SI-DENORM-SLOWFMAF: v_mul_f32_e32 [[TMP:v[0-9]+]], [[A]], [[B]]
; SI-DENORM-SLOWFMAF-DAG: v_sub_f32_e32 [[RESULT0:v[0-9]+]], [[TMP]], [[C]]
; SI-DENORM-SLOWFMAF-DAG: v_sub_f32_e32 [[RESULT1:v[0-9]+]], [[TMP]], [[D]]

; SI-DAG: buffer_store_dword [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dword [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4{{$}}
; SI: s_endpgm
define amdgpu_kernel void @combine_to_mad_fsub_0_f32_2use(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr float, float addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr float, float addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr float, float addrspace(1)* %gep.out.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0
  %b = load volatile float, float addrspace(1)* %gep.1
  %c = load volatile float, float addrspace(1)* %gep.2
  %d = load volatile float, float addrspace(1)* %gep.3

  %mul = fmul float %a, %b
  %fma0 = fsub float %mul, %c
  %fma1 = fsub float %mul, %d
  store volatile float %fma0, float addrspace(1)* %gep.out.0
  store volatile float %fma1, float addrspace(1)* %gep.out.1
  ret void
}

; (fsub x, (fmul y, z)) -> (fma (fneg y), z, x)
; FUNC-LABEL: {{^}}combine_to_mad_fsub_1_f32:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4 glc{{$}}
; SI-DAG: buffer_load_dword [[C:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}

; SI-STD: v_mad_f32 [[RESULT:v[0-9]+]], -[[A]], [[B]], [[C]]
; SI-DENORM-FASTFMAF: v_fma_f32 [[RESULT:v[0-9]+]], -[[A]], [[B]], [[C]]

; SI-DENORM-SLOWFMAF: v_mul_f32_e32 [[TMP:v[0-9]+]], [[A]], [[B]]
; SI-DENORM-SLOWFMAF: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[C]], [[TMP]]

; SI: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @combine_to_mad_fsub_1_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr float, float addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load volatile float, float addrspace(1)* %gep.0
  %b = load volatile float, float addrspace(1)* %gep.1
  %c = load volatile float, float addrspace(1)* %gep.2

  %mul = fmul float %a, %b
  %fma = fsub float %c, %mul
  store float %fma, float addrspace(1)* %gep.out
  ret void
}

; (fsub x, (fmul y, z)) -> (fma (fneg y), z, x)
; FUNC-LABEL: {{^}}combine_to_mad_fsub_1_f32_2use:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4 glc{{$}}
; SI-DAG: buffer_load_dword [[C:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dword [[D:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:12 glc{{$}}

; SI-STD-DAG: v_mad_f32 [[RESULT0:v[0-9]+]], -[[A]], [[B]], [[C]]
; SI-STD-DAG: v_mad_f32 [[RESULT1:v[0-9]+]], -[[A]], [[B]], [[D]]

; SI-DENORM-FASTFMAF-DAG: v_fma_f32 [[RESULT0:v[0-9]+]], -[[A]], [[B]], [[C]]
; SI-DENORM-FASTFMAF-DAG: v_fma_f32 [[RESULT1:v[0-9]+]], -[[A]], [[B]], [[D]]

; SI-DENORM-SLOWFMAF: v_mul_f32_e32 [[TMP:v[0-9]+]], [[A]], [[B]]
; SI-DENORM-SLOWFMAF-DAG: v_sub_f32_e32 [[RESULT0:v[0-9]+]], [[C]], [[TMP]]
; SI-DENORM-SLOWFMAF-DAG: v_sub_f32_e32 [[RESULT1:v[0-9]+]],  [[D]], [[TMP]]

; SI-DAG: buffer_store_dword [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dword [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4{{$}}
; SI: s_endpgm
define amdgpu_kernel void @combine_to_mad_fsub_1_f32_2use(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr float, float addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr float, float addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr float, float addrspace(1)* %gep.out.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0
  %b = load volatile float, float addrspace(1)* %gep.1
  %c = load volatile float, float addrspace(1)* %gep.2
  %d = load volatile float, float addrspace(1)* %gep.3

  %mul = fmul float %a, %b
  %fma0 = fsub float %c, %mul
  %fma1 = fsub float %d, %mul
  store volatile float %fma0, float addrspace(1)* %gep.out.0
  store volatile float %fma1, float addrspace(1)* %gep.out.1
  ret void
}

; (fsub (fneg (fmul x, y)), z) -> (fma (fneg x), y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_mad_fsub_2_f32:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4 glc{{$}}
; SI-DAG: buffer_load_dword [[C:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}

; SI-STD: v_mad_f32 [[RESULT:v[0-9]+]], [[A]], -[[B]], -[[C]]

; SI-DENORM-FASTFMAF: v_fma_f32 [[RESULT:v[0-9]+]], -[[A]], [[B]], -[[C]]

; SI-DENORM-SLOWFMAF: v_mul_f32_e64 [[TMP:v[0-9]+]], [[A]], -[[B]]
; SI-DENORM-SLOWFMAF: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[TMP]], [[C]]

; SI: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @combine_to_mad_fsub_2_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr float, float addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load volatile float, float addrspace(1)* %gep.0
  %b = load volatile float, float addrspace(1)* %gep.1
  %c = load volatile float, float addrspace(1)* %gep.2

  %mul = fmul float %a, %b
  %mul.neg = fneg float %mul
  %fma = fsub float %mul.neg, %c

  store float %fma, float addrspace(1)* %gep.out
  ret void
}

; (fsub (fneg (fmul x, y)), z) -> (fma (fneg x), y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_mad_fsub_2_f32_2uses_neg:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4 glc{{$}}
; SI-DAG: buffer_load_dword [[C:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dword [[D:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:12 glc{{$}}

; SI-STD-DAG: v_mad_f32 [[RESULT0:v[0-9]+]], [[A]], -[[B]], -[[C]]
; SI-STD-DAG: v_mad_f32 [[RESULT1:v[0-9]+]], [[A]], -[[B]], -[[D]]

; SI-DENORM-FASTFMAF-DAG: v_fma_f32 [[RESULT0:v[0-9]+]], -[[A]], [[B]], -[[C]]
; SI-DENORM-FASTFMAF-DAG: v_fma_f32 [[RESULT1:v[0-9]+]], -[[A]], [[B]], -[[D]]

; SI-DENORM-SLOWFMAF: v_mul_f32_e64 [[TMP:v[0-9]+]], [[A]], -[[B]]
; SI-DENORM-SLOWFMAF-DAG: v_sub_f32_e32 [[RESULT0:v[0-9]+]], [[TMP]], [[C]]
; SI-DENORM-SLOWFMAF-DAG: v_sub_f32_e32 [[RESULT1:v[0-9]+]],  [[TMP]], [[D]]

; SI-DAG: buffer_store_dword [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dword [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4{{$}}
; SI: s_endpgm
define amdgpu_kernel void @combine_to_mad_fsub_2_f32_2uses_neg(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr float, float addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr float, float addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr float, float addrspace(1)* %gep.out.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0
  %b = load volatile float, float addrspace(1)* %gep.1
  %c = load volatile float, float addrspace(1)* %gep.2
  %d = load volatile float, float addrspace(1)* %gep.3

  %mul = fmul float %a, %b
  %mul.neg = fneg float %mul
  %fma0 = fsub float %mul.neg, %c
  %fma1 = fsub float %mul.neg, %d

  store volatile float %fma0, float addrspace(1)* %gep.out.0
  store volatile float %fma1, float addrspace(1)* %gep.out.1
  ret void
}

; (fsub (fneg (fmul x, y)), z) -> (fma (fneg x), y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_mad_fsub_2_f32_2uses_mul:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4 glc{{$}}
; SI-DAG: buffer_load_dword [[C:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dword [[D:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:12 glc{{$}}

; SI-STD-DAG: v_mad_f32 [[RESULT0:v[0-9]+]], -[[A]], [[B]], -[[C]]
; SI-STD-DAG: v_mad_f32 [[RESULT1:v[0-9]+]], [[A]], [[B]], -[[D]]

; SI-DENORM-FASTFMAF-DAG: v_fma_f32 [[RESULT0:v[0-9]+]], -[[A]], [[B]], -[[C]]
; SI-DENORM-FASTFMAF-DAG: v_fma_f32 [[RESULT1:v[0-9]+]], [[A]], [[B]], -[[D]]

; SI-DENORM-SLOWFMAF: v_mul_f32_e32 [[TMP:v[0-9]+]], [[A]], [[B]]
; SI-DENORM-SLOWFMAF-DAG: v_sub_f32_e64 [[RESULT0:v[0-9]+]], -[[TMP]], [[C]]
; SI-DENORM-SLOWFMAF-DAG: v_sub_f32_e32 [[RESULT1:v[0-9]+]], [[TMP]], [[D]]

; SI-DAG: buffer_store_dword [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dword [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4{{$}}
; SI: s_endpgm
define amdgpu_kernel void @combine_to_mad_fsub_2_f32_2uses_mul(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr float, float addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr float, float addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr float, float addrspace(1)* %gep.out.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0
  %b = load volatile float, float addrspace(1)* %gep.1
  %c = load volatile float, float addrspace(1)* %gep.2
  %d = load volatile float, float addrspace(1)* %gep.3

  %mul = fmul float %a, %b
  %mul.neg = fneg float %mul
  %fma0 = fsub float %mul.neg, %c
  %fma1 = fsub float %mul, %d

  store volatile float %fma0, float addrspace(1)* %gep.out.0
  store volatile float %fma1, float addrspace(1)* %gep.out.1
  ret void
}

; fold (fsub (fma x, y, (fmul u, v)), z) -> (fma x, y (fma u, v, (fneg z)))

; FUNC-LABEL: {{^}}aggressive_combine_to_mad_fsub_0_f32:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4 glc{{$}}
; SI-DAG: buffer_load_dword [[C:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dword [[D:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:12 glc{{$}}
; SI-DAG: buffer_load_dword [[E:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16 glc{{$}}

; SI-STD-SAFE: v_mul_f32_e32 [[TMP0:v[0-9]+]], [[D]], [[E]]
; SI-STD-SAFE: v_fma_f32 [[TMP1:v[0-9]+]], [[A]], [[B]], [[TMP0]]
; SI-STD-SAFE: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[TMP1]], [[C]]

; SI-STD-UNSAFE: v_mad_f32 [[RESULT:v[0-9]+]], [[D]], [[E]], -[[C]]
; SI-STD-UNSAFE: v_mac_f32_e32 [[RESULT]], [[A]], [[B]]

; SI-DENORM: v_mul_f32_e32 [[TMP0:v[0-9]+]], [[D]], [[E]]
; SI-DENORM: v_fma_f32 [[TMP1:v[0-9]+]], [[A]], [[B]], [[TMP0]]
; SI-DENORM: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[TMP1]], [[C]]

; SI: buffer_store_dword [[RESULT]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
define amdgpu_kernel void @aggressive_combine_to_mad_fsub_0_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr float, float addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr float, float addrspace(1)* %gep.0, i32 3
  %gep.4 = getelementptr float, float addrspace(1)* %gep.0, i32 4
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %x = load volatile float, float addrspace(1)* %gep.0
  %y = load volatile float, float addrspace(1)* %gep.1
  %z = load volatile float, float addrspace(1)* %gep.2
  %u = load volatile float, float addrspace(1)* %gep.3
  %v = load volatile float, float addrspace(1)* %gep.4

  %tmp0 = fmul float %u, %v
  %tmp1 = call float @llvm.fma.f32(float %x, float %y, float %tmp0) #0
  %tmp2 = fsub float %tmp1, %z

  store float %tmp2, float addrspace(1)* %gep.out
  ret void
}

; fold (fsub x, (fma y, z, (fmul u, v)))
;   -> (fma (fneg y), z, (fma (fneg u), v, x))

; FUNC-LABEL: {{^}}aggressive_combine_to_mad_fsub_1_f32:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4 glc{{$}}
; SI-DAG: buffer_load_dword [[C:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dword [[D:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:12 glc{{$}}
; SI-DAG: buffer_load_dword [[E:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16 glc{{$}}

; SI-STD: v_mul_f32_e32 [[TMP0:v[0-9]+]], [[D]], [[E]]
; SI-STD: v_fma_f32 [[TMP1:v[0-9]+]], [[B]], [[C]], [[TMP0]]
; SI-STD: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[A]], [[TMP1]]

; SI-DENORM: v_mul_f32_e32 [[TMP0:v[0-9]+]], [[D]], [[E]]
; SI-DENORM: v_fma_f32 [[TMP1:v[0-9]+]], [[B]], [[C]], [[TMP0]]
; SI-DENORM: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[A]], [[TMP1]]

; SI: buffer_store_dword [[RESULT]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: s_endpgm
define amdgpu_kernel void @aggressive_combine_to_mad_fsub_1_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr float, float addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr float, float addrspace(1)* %gep.0, i32 3
  %gep.4 = getelementptr float, float addrspace(1)* %gep.0, i32 4
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %x = load volatile float, float addrspace(1)* %gep.0
  %y = load volatile float, float addrspace(1)* %gep.1
  %z = load volatile float, float addrspace(1)* %gep.2
  %u = load volatile float, float addrspace(1)* %gep.3
  %v = load volatile float, float addrspace(1)* %gep.4

  %tmp0 = fmul float %u, %v
  %tmp1 = call float @llvm.fma.f32(float %y, float %z, float %tmp0) #0
  %tmp2 = fsub float %x, %tmp1

  store float %tmp2, float addrspace(1)* %gep.out
  ret void
}

; fold (fsub (fma x, y, (fmul u, v)), z) -> (fma x, y (fma u, v, (fneg z)))

; FUNC-LABEL: {{^}}aggressive_combine_to_mad_fsub_2_f32:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4 glc{{$}}
; SI-DAG: buffer_load_dword [[C:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dword [[D:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:12 glc{{$}}
; SI-DAG: buffer_load_dword [[E:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16 glc{{$}}

; SI-STD-SAFE: v_mul_f32_e32 [[TMP0:v[0-9]+]], [[D]], [[E]]
; SI-STD-SAFE: v_mac_f32_e32 [[TMP0]], [[A]], [[B]]
; SI-STD-SAFE: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[TMP0]], [[C]]

; SI-STD-UNSAFE: v_mad_f32 [[RESULT:v[0-9]+]], [[D]], [[E]], -[[C]]
; SI-STD-UNSAFE: v_mac_f32_e32 [[RESULT]], [[A]], [[B]]

; SI-DENORM-FASTFMAF: v_mul_f32_e32 [[TMP0:v[0-9]+]], [[D]], [[E]]
; SI-DENORM-FASTFMAF: v_fma_f32 [[TMP1:v[0-9]+]], [[A]], [[B]], [[TMP0]]
; SI-DENORM-FASTFMAF: v_sub_f32_e32 [[RESULT:v[0-9]+]],  [[TMP1]], [[C]]

; SI-DENORM-SLOWFMAF-DAG: v_mul_f32_e32 [[TMP0:v[0-9]+]], [[D]], [[E]]
; SI-DENORM-SLOWFMAF-DAG: v_mul_f32_e32 [[TMP1:v[0-9]+]], [[A]], [[B]]
; SI-DENORM-SLOWFMAF: v_add_f32_e32 [[TMP2:v[0-9]+]], [[TMP1]], [[TMP0]]
; SI-DENORM-SLOWFMAF: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[TMP2]], [[C]]

; SI: buffer_store_dword [[RESULT]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: s_endpgm
define amdgpu_kernel void @aggressive_combine_to_mad_fsub_2_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr float, float addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr float, float addrspace(1)* %gep.0, i32 3
  %gep.4 = getelementptr float, float addrspace(1)* %gep.0, i32 4
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %x = load volatile float, float addrspace(1)* %gep.0
  %y = load volatile float, float addrspace(1)* %gep.1
  %z = load volatile float, float addrspace(1)* %gep.2
  %u = load volatile float, float addrspace(1)* %gep.3
  %v = load volatile float, float addrspace(1)* %gep.4

  %tmp0 = fmul float %u, %v
  %tmp1 = call float @llvm.fmuladd.f32(float %x, float %y, float %tmp0) #0
  %tmp2 = fsub float %tmp1, %z

  store float %tmp2, float addrspace(1)* %gep.out
  ret void
}

; fold (fsub x, (fmuladd y, z, (fmul u, v)))
;   -> (fmuladd (fneg y), z, (fmuladd (fneg u), v, x))

; FUNC-LABEL: {{^}}aggressive_combine_to_mad_fsub_3_f32:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4 glc{{$}}
; SI-DAG: buffer_load_dword [[C:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dword [[D:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:12 glc{{$}}
; SI-DAG: buffer_load_dword [[E:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16 glc{{$}}

; SI-STD-SAFE: v_mul_f32_e32 [[TMP0:v[0-9]+]], [[D]], [[E]]
; SI-STD-SAFE: v_mac_f32_e32 [[TMP0]], [[B]], [[C]]
; SI-STD-SAFE: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[A]], [[TMP0]]

; SI-STD-UNSAFE: v_mad_f32 [[TMP:v[0-9]+]], -[[D]], [[E]], [[A]]
; SI-STD-UNSAFE: v_mad_f32 [[RESULT:v[0-9]+]], -[[B]], [[C]], [[TMP]]

; SI-DENORM-FASTFMAF: v_mul_f32_e32 [[TMP0:v[0-9]+]], [[D]], [[E]]
; SI-DENORM-FASTFMAF: v_fma_f32 [[TMP1:v[0-9]+]], [[B]], [[C]], [[TMP0]]
; SI-DENORM-FASTFMAF: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[A]], [[TMP1]]

; SI-DENORM-SLOWFMAF-DAG: v_mul_f32_e32 [[TMP0:v[0-9]+]], [[D]], [[E]]
; SI-DENORM-SLOWFMAF-DAG: v_mul_f32_e32 [[TMP1:v[0-9]+]], [[B]], [[C]]
; SI-DENORM-SLOWFMAF: v_add_f32_e32 [[TMP2:v[0-9]+]], [[TMP1]], [[TMP0]]
; SI-DENORM-SLOWFMAF: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[A]], [[TMP2]]

; SI: buffer_store_dword [[RESULT]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: s_endpgm
define amdgpu_kernel void @aggressive_combine_to_mad_fsub_3_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr float, float addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr float, float addrspace(1)* %gep.0, i32 3
  %gep.4 = getelementptr float, float addrspace(1)* %gep.0, i32 4
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %x = load volatile float, float addrspace(1)* %gep.0
  %y = load volatile float, float addrspace(1)* %gep.1
  %z = load volatile float, float addrspace(1)* %gep.2
  %u = load volatile float, float addrspace(1)* %gep.3
  %v = load volatile float, float addrspace(1)* %gep.4

  ; nsz flag is needed since this combine may change sign of zero
  %tmp0 = fmul nsz float %u, %v
  %tmp1 = call nsz float @llvm.fmuladd.f32(float %y, float %z, float %tmp0) #0
  %tmp2 = fsub nsz float %x, %tmp1

  store float %tmp2, float addrspace(1)* %gep.out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

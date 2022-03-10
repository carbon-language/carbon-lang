; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -denormal-fp-math-f32=preserve-sign -verify-machineinstrs -fp-contract=fast < %s | FileCheck -enable-var-scope -check-prefix=SI-NOFMA -check-prefix=SI-SAFE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=verde -denormal-fp-math-f32=preserve-sign -verify-machineinstrs -fp-contract=fast < %s | FileCheck -enable-var-scope -check-prefix=SI-NOFMA -check-prefix=SI-SAFE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -denormal-fp-math-f32=ieee -verify-machineinstrs -fp-contract=fast -enable-no-infs-fp-math -enable-unsafe-fp-math < %s | FileCheck -enable-var-scope -check-prefix=SI-FMA -check-prefix=SI-UNSAFE -check-prefix=SI -check-prefix=FUNC %s

; FIXME: Remove enable-unsafe-fp-math in RUN line and add flags to IR instrs

; Note: The SI-FMA conversions of type x * (y + 1) --> x * y + x would be
; beneficial even without fp32 denormals, but they do require no-infs-fp-math
; for correctness.

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare double @llvm.fabs.f64(double) #0
declare double @llvm.fma.f64(double, double, double) #0
declare float @llvm.fma.f32(float, float, float) #0
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>) #0

; (fadd (fmul x, y), z) -> (fma x, y, z)
; FUNC-LABEL: {{^}}combine_to_fma_f64_0:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16 glc{{$}}
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[C]]
; SI: buffer_store_dwordx2 [[RESULT]]
define amdgpu_kernel void @combine_to_fma_f64_0(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2

  %mul = fmul double %a, %b
  %fma = fadd double %mul, %c
  store double %fma, double addrspace(1)* %gep.out
  ret void
}

; (fadd (fmul x, y), z) -> (fma x, y, z)
; FUNC-LABEL: {{^}}combine_to_fma_f64_0_2use:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[D:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24 glc{{$}}
; SI-DAG: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[C]]
; SI-DAG: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[D]]
; SI-DAG: buffer_store_dwordx2 [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dwordx2 [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI: s_endpgm
define amdgpu_kernel void @combine_to_fma_f64_0_2use(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr double, double addrspace(1)* %gep.out.0, i32 1

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2
  %d = load volatile double, double addrspace(1)* %gep.3

  %mul = fmul double %a, %b
  %fma0 = fadd double %mul, %c
  %fma1 = fadd double %mul, %d
  store volatile double %fma0, double addrspace(1)* %gep.out.0
  store volatile double %fma1, double addrspace(1)* %gep.out.1
  ret void
}

; (fadd x, (fmul y, z)) -> (fma y, z, x)
; FUNC-LABEL: {{^}}combine_to_fma_f64_1:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16 glc{{$}}
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[C]]
; SI: buffer_store_dwordx2 [[RESULT]]
define amdgpu_kernel void @combine_to_fma_f64_1(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2

  %mul = fmul double %a, %b
  %fma = fadd double %c, %mul
  store double %fma, double addrspace(1)* %gep.out
  ret void
}

; (fsub (fmul x, y), z) -> (fma x, y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_fma_fsub_0_f64:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16 glc{{$}}
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], -[[C]]
; SI: buffer_store_dwordx2 [[RESULT]]
define amdgpu_kernel void @combine_to_fma_fsub_0_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2

  %mul = fmul double %a, %b
  %fma = fsub double %mul, %c
  store double %fma, double addrspace(1)* %gep.out
  ret void
}

; (fsub (fmul x, y), z) -> (fma x, y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_fma_fsub_f64_0_2use:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[D:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24 glc{{$}}
; SI-DAG: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], -[[C]]
; SI-DAG: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], -[[D]]
; SI-DAG: buffer_store_dwordx2 [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dwordx2 [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI: s_endpgm
define amdgpu_kernel void @combine_to_fma_fsub_f64_0_2use(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr double, double addrspace(1)* %gep.out.0, i32 1

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2
  %d = load volatile double, double addrspace(1)* %gep.3

  %mul = fmul double %a, %b
  %fma0 = fsub double %mul, %c
  %fma1 = fsub double %mul, %d
  store volatile double %fma0, double addrspace(1)* %gep.out.0
  store volatile double %fma1, double addrspace(1)* %gep.out.1
  ret void
}

; (fsub x, (fmul y, z)) -> (fma (fneg y), z, x)
; FUNC-LABEL: {{^}}combine_to_fma_fsub_1_f64:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16 glc{{$}}
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], [[C]]
; SI: buffer_store_dwordx2 [[RESULT]]
define amdgpu_kernel void @combine_to_fma_fsub_1_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2

  %mul = fmul double %a, %b
  %fma = fsub double %c, %mul
  store double %fma, double addrspace(1)* %gep.out
  ret void
}

; (fsub x, (fmul y, z)) -> (fma (fneg y), z, x)
; FUNC-LABEL: {{^}}combine_to_fma_fsub_1_f64_2use:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[D:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24 glc{{$}}
; SI-DAG: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], [[C]]
; SI-DAG: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], [[D]]
; SI-DAG: buffer_store_dwordx2 [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dwordx2 [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI: s_endpgm
define amdgpu_kernel void @combine_to_fma_fsub_1_f64_2use(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr double, double addrspace(1)* %gep.out.0, i32 1

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2
  %d = load volatile double, double addrspace(1)* %gep.3

  %mul = fmul double %a, %b
  %fma0 = fsub double %c, %mul
  %fma1 = fsub double %d, %mul
  store volatile double %fma0, double addrspace(1)* %gep.out.0
  store volatile double %fma1, double addrspace(1)* %gep.out.1
  ret void
}

; (fsub (fneg (fmul x, y)), z) -> (fma (fneg x), y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_fma_fsub_2_f64:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16 glc{{$}}
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], -[[C]]
; SI: buffer_store_dwordx2 [[RESULT]]
define amdgpu_kernel void @combine_to_fma_fsub_2_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2

  %mul = fmul double %a, %b
  %mul.neg = fsub double -0.0, %mul
  %fma = fsub double %mul.neg, %c

  store double %fma, double addrspace(1)* %gep.out
  ret void
}

; (fsub (fneg (fmul x, y)), z) -> (fma (fneg x), y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_fma_fsub_2_f64_2uses_neg:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[D:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24 glc{{$}}
; SI-DAG: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], -[[C]]
; SI-DAG: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], -[[D]]
; SI-DAG: buffer_store_dwordx2 [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dwordx2 [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI: s_endpgm
define amdgpu_kernel void @combine_to_fma_fsub_2_f64_2uses_neg(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr double, double addrspace(1)* %gep.out.0, i32 1

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2
  %d = load volatile double, double addrspace(1)* %gep.3

  %mul = fmul double %a, %b
  %mul.neg = fsub double -0.0, %mul
  %fma0 = fsub double %mul.neg, %c
  %fma1 = fsub double %mul.neg, %d

  store volatile double %fma0, double addrspace(1)* %gep.out.0
  store volatile double %fma1, double addrspace(1)* %gep.out.1
  ret void
}

; (fsub (fneg (fmul x, y)), z) -> (fma (fneg x), y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_fma_fsub_2_f64_2uses_mul:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[D:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24 glc{{$}}
; SI-DAG: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], -[[C]]
; SI-DAG: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], -[[D]]
; SI-DAG: buffer_store_dwordx2 [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dwordx2 [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI: s_endpgm
define amdgpu_kernel void @combine_to_fma_fsub_2_f64_2uses_mul(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr double, double addrspace(1)* %gep.out.0, i32 1

  %a = load volatile double, double addrspace(1)* %gep.0
  %b = load volatile double, double addrspace(1)* %gep.1
  %c = load volatile double, double addrspace(1)* %gep.2
  %d = load volatile double, double addrspace(1)* %gep.3

  %mul = fmul double %a, %b
  %mul.neg = fsub double -0.0, %mul
  %fma0 = fsub double %mul.neg, %c
  %fma1 = fsub double %mul, %d

  store volatile double %fma0, double addrspace(1)* %gep.out.0
  store volatile double %fma1, double addrspace(1)* %gep.out.1
  ret void
}

; fold (fsub (fma x, y, (fmul u, v)), z) -> (fma x, y (fma u, v, (fneg z)))

; FUNC-LABEL: {{^}}aggressive_combine_to_fma_fsub_0_f64:
; SI-DAG: buffer_load_dwordx2 [[X:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[Y:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[Z:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[U:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[V:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:32 glc{{$}}

; SI-SAFE: v_mul_f64 [[TMP0:v\[[0-9]+:[0-9]+\]]], [[U]], [[V]]
; SI-SAFE: v_fma_f64 [[TMP1:v\[[0-9]+:[0-9]+\]]], [[X]], [[Y]], [[TMP0]]
; SI-SAFE: v_add_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[TMP1]], -[[Z]]

; SI-UNSAFE: v_fma_f64 [[FMA0:v\[[0-9]+:[0-9]+\]]], [[U]], [[V]], -[[Z]]
; SI-UNSAFE: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[X]], [[Y]], [[FMA0]]

; SI: buffer_store_dwordx2 [[RESULT]]
define amdgpu_kernel void @aggressive_combine_to_fma_fsub_0_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.4 = getelementptr double, double addrspace(1)* %gep.0, i32 4
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %x = load volatile double, double addrspace(1)* %gep.0
  %y = load volatile double, double addrspace(1)* %gep.1
  %z = load volatile double, double addrspace(1)* %gep.2
  %u = load volatile double, double addrspace(1)* %gep.3
  %v = load volatile double, double addrspace(1)* %gep.4

  %tmp0 = fmul double %u, %v
  %tmp1 = call double @llvm.fma.f64(double %x, double %y, double %tmp0) #0
  %tmp2 = fsub double %tmp1, %z

  store double %tmp2, double addrspace(1)* %gep.out
  ret void
}

; fold (fsub x, (fma y, z, (fmul u, v)))
;   -> (fma (fneg y), z, (fma (fneg u), v, x))

; FUNC-LABEL: {{^}}aggressive_combine_to_fma_fsub_1_f64:
; SI-DAG: buffer_load_dwordx2 [[X:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[Y:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[Z:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[U:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24 glc{{$}}
; SI-DAG: buffer_load_dwordx2 [[V:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:32 glc{{$}}

; SI-SAFE: v_mul_f64 [[TMP0:v\[[0-9]+:[0-9]+\]]], [[U]], [[V]]
; SI-SAFE: v_fma_f64 [[TMP1:v\[[0-9]+:[0-9]+\]]], [[Y]], [[Z]], [[TMP0]]
; SI-SAFE: v_add_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[X]], -[[TMP1]]

; SI-UNSAFE: v_fma_f64 [[FMA0:v\[[0-9]+:[0-9]+\]]], -[[U]], [[V]], [[X]]
; SI-UNSAFE: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], -[[Y]], [[Z]], [[FMA0]]

; SI: buffer_store_dwordx2 [[RESULT]]
define amdgpu_kernel void @aggressive_combine_to_fma_fsub_1_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.4 = getelementptr double, double addrspace(1)* %gep.0, i32 4
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %x = load volatile double, double addrspace(1)* %gep.0
  %y = load volatile double, double addrspace(1)* %gep.1
  %z = load volatile double, double addrspace(1)* %gep.2
  %u = load volatile double, double addrspace(1)* %gep.3
  %v = load volatile double, double addrspace(1)* %gep.4

  ; nsz flag is needed since this combine may change sign of zero
  %tmp0 = fmul nsz double %u, %v
  %tmp1 = call nsz double @llvm.fma.f64(double %y, double %z, double %tmp0) #0
  %tmp2 = fsub nsz double %x, %tmp1

  store double %tmp2, double addrspace(1)* %gep.out
  ret void
}

;
; Patterns (+ fneg variants): mul(add(1.0,x),y), mul(sub(1.0,x),y), mul(sub(x,1.0),y)
;

; FUNC-LABEL: {{^}}test_f32_mul_add_x_one_y:
; SI-NOFMA: v_add_f32_e32 [[VS:v[0-9]]], 1.0, [[VX:v[0-9]]]
; SI-NOFMA: v_mul_f32_e32 {{v[0-9]}}, [[VS]], [[VY:v[0-9]]]
;
; SI-FMA: v_fma_f32 {{v[0-9]}}, [[VX:v[0-9]]], [[VY:v[0-9]]], [[VY:v[0-9]]]
define amdgpu_kernel void @test_f32_mul_add_x_one_y(float addrspace(1)* %out,
                                        float addrspace(1)* %in1,
                                        float addrspace(1)* %in2) {
  %x = load volatile float, float addrspace(1)* %in1
  %y = load volatile float, float addrspace(1)* %in2
  %a = fadd float %x, 1.0
  %m = fmul float %a, %y
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_y_add_x_one:
; SI-NOFMA: v_add_f32_e32 [[VS:v[0-9]]], 1.0, [[VX:v[0-9]]]
; SI-NOFMA: v_mul_f32_e32 {{v[0-9]}}, [[VY:v[0-9]]], [[VS]]
;
; SI-FMA: v_fma_f32 {{v[0-9]}}, [[VX:v[0-9]]], [[VY:v[0-9]]], [[VY:v[0-9]]]
define amdgpu_kernel void @test_f32_mul_y_add_x_one(float addrspace(1)* %out,
                                        float addrspace(1)* %in1,
                                        float addrspace(1)* %in2) {
  %x = load volatile float, float addrspace(1)* %in1
  %y = load volatile float, float addrspace(1)* %in2
  %a = fadd float %x, 1.0
  %m = fmul float %y, %a
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_add_x_negone_y:
; SI-NOFMA: v_add_f32_e32 [[VS:v[0-9]]], -1.0, [[VX:v[0-9]]]
; SI-NOFMA: v_mul_f32_e32 {{v[0-9]}}, [[VS]], [[VY:v[0-9]]]
;
; SI-FMA: v_fma_f32 {{v[0-9]}}, [[VX:v[0-9]]], [[VY:v[0-9]]], -[[VY:v[0-9]]]
define amdgpu_kernel void @test_f32_mul_add_x_negone_y(float addrspace(1)* %out,
                                           float addrspace(1)* %in1,
                                           float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %a = fadd float %x, -1.0
  %m = fmul float %a, %y
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_y_add_x_negone:
; SI-NOFMA: v_add_f32_e32 [[VS:v[0-9]]], -1.0, [[VX:v[0-9]]]
; SI-NOFMA: v_mul_f32_e32 {{v[0-9]}}, [[VY:v[0-9]]], [[VS]]
;
; SI-FMA: v_fma_f32 {{v[0-9]}}, [[VX:v[0-9]]], [[VY:v[0-9]]], -[[VY:v[0-9]]]
define amdgpu_kernel void @test_f32_mul_y_add_x_negone(float addrspace(1)* %out,
                                           float addrspace(1)* %in1,
                                           float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %a = fadd float %x, -1.0
  %m = fmul float %y, %a
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_sub_one_x_y:
; SI-NOFMA: v_sub_f32_e32 [[VS:v[0-9]]], 1.0, [[VX:v[0-9]]]
; SI-NOFMA: v_mul_f32_e32 {{v[0-9]}}, [[VS]], [[VY:v[0-9]]]
;
; SI-FMA: v_fma_f32 {{v[0-9]}}, -[[VX:v[0-9]]], [[VY:v[0-9]]], [[VY:v[0-9]]]
define amdgpu_kernel void @test_f32_mul_sub_one_x_y(float addrspace(1)* %out,
                                        float addrspace(1)* %in1,
                                        float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %s = fsub float 1.0, %x
  %m = fmul float %s, %y
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_y_sub_one_x:
; SI-NOFMA: v_sub_f32_e32 [[VS:v[0-9]]], 1.0, [[VX:v[0-9]]]
; SI-NOFMA: v_mul_f32_e32 {{v[0-9]}}, [[VY:v[0-9]]], [[VS]]
;
; SI-FMA: v_fma_f32 {{v[0-9]}}, -[[VX:v[0-9]]], [[VY:v[0-9]]], [[VY:v[0-9]]]
define amdgpu_kernel void @test_f32_mul_y_sub_one_x(float addrspace(1)* %out,
                                        float addrspace(1)* %in1,
                                        float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %s = fsub float 1.0, %x
  %m = fmul float %y, %s
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_sub_negone_x_y:
; SI-NOFMA: v_sub_f32_e32 [[VS:v[0-9]]], -1.0, [[VX:v[0-9]]]
; SI-NOFMA: v_mul_f32_e32 {{v[0-9]}}, [[VS]], [[VY:v[0-9]]]
;
; SI-FMA: v_fma_f32 {{v[0-9]}}, -[[VX:v[0-9]]], [[VY:v[0-9]]], -[[VY:v[0-9]]]
define amdgpu_kernel void @test_f32_mul_sub_negone_x_y(float addrspace(1)* %out,
                                           float addrspace(1)* %in1,
                                           float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %s = fsub float -1.0, %x
  %m = fmul float %s, %y
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_y_sub_negone_x:
; SI-NOFMA: v_sub_f32_e32 [[VS:v[0-9]]], -1.0, [[VX:v[0-9]]]
; SI-NOFMA: v_mul_f32_e32 {{v[0-9]}}, [[VY:v[0-9]]], [[VS]]
;
; SI-FMA: v_fma_f32 {{v[0-9]}}, -[[VX:v[0-9]]], [[VY:v[0-9]]], -[[VY:v[0-9]]]
define amdgpu_kernel void @test_f32_mul_y_sub_negone_x(float addrspace(1)* %out,
                                         float addrspace(1)* %in1,
                                         float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %s = fsub float -1.0, %x
  %m = fmul float %y, %s
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_sub_x_one_y:
; SI-NOFMA: v_add_f32_e32 [[VS:v[0-9]]], -1.0, [[VX:v[0-9]]]
; SI-NOFMA: v_mul_f32_e32 {{v[0-9]}}, [[VS]], [[VY:v[0-9]]]
;
; SI-FMA: v_fma_f32 {{v[0-9]}}, [[VX:v[0-9]]], [[VY:v[0-9]]], -[[VY:v[0-9]]]
define amdgpu_kernel void @test_f32_mul_sub_x_one_y(float addrspace(1)* %out,
                                        float addrspace(1)* %in1,
                                        float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %s = fsub float %x, 1.0
  %m = fmul float %s, %y
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_y_sub_x_one:
; SI-NOFMA: v_add_f32_e32 [[VS:v[0-9]]], -1.0, [[VX:v[0-9]]]
; SI-NOFMA: v_mul_f32_e32 {{v[0-9]}}, [[VY:v[0-9]]], [[VS]]
;
; SI-FMA: v_fma_f32 {{v[0-9]}}, [[VX:v[0-9]]], [[VY:v[0-9]]], -[[VY:v[0-9]]]
define amdgpu_kernel void @test_f32_mul_y_sub_x_one(float addrspace(1)* %out,
                                      float addrspace(1)* %in1,
                                      float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %s = fsub float %x, 1.0
  %m = fmul float %y, %s
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_sub_x_negone_y:
; SI-NOFMA: v_add_f32_e32 [[VS:v[0-9]]], 1.0, [[VX:v[0-9]]]
; SI-NOFMA: v_mul_f32_e32 {{v[0-9]}}, [[VS]], [[VY:v[0-9]]]
;
; SI-FMA: v_fma_f32 {{v[0-9]}}, [[VX:v[0-9]]], [[VY:v[0-9]]], [[VY:v[0-9]]]
define amdgpu_kernel void @test_f32_mul_sub_x_negone_y(float addrspace(1)* %out,
                                         float addrspace(1)* %in1,
                                         float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %s = fsub float %x, -1.0
  %m = fmul float %s, %y
  store float %m, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f32_mul_y_sub_x_negone:
; SI-NOFMA: v_add_f32_e32 [[VS:v[0-9]]], 1.0, [[VX:v[0-9]]]
; SI-NOFMA: v_mul_f32_e32 {{v[0-9]}}, [[VY:v[0-9]]], [[VS]]
;
; SI-FMA: v_fma_f32 {{v[0-9]}}, [[VX:v[0-9]]], [[VY:v[0-9]]], [[VY:v[0-9]]]
define amdgpu_kernel void @test_f32_mul_y_sub_x_negone(float addrspace(1)* %out,
                                         float addrspace(1)* %in1,
                                         float addrspace(1)* %in2) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %s = fsub float %x, -1.0
  %m = fmul float %y, %s
  store float %m, float addrspace(1)* %out
  ret void
}

;
; Interpolation Patterns: add(mul(x,t),mul(sub(1.0,t),y))
;

; FUNC-LABEL: {{^}}test_f32_interp:
; SI-NOFMA: v_sub_f32_e32 [[VT1:v[0-9]]], 1.0, [[VT:v[0-9]]]
; SI-NOFMA: v_mul_f32_e32 [[VTY:v[0-9]]], [[VY:v[0-9]]], [[VT1]]
; SI-NOFMA: v_mac_f32_e32 [[VTY]], [[VX:v[0-9]]], [[VT]]
;
; SI-FMA: v_fma_f32 [[VR:v[0-9]]], -[[VT:v[0-9]]], [[VY:v[0-9]]], [[VY]]
; SI-FMA: v_fma_f32 {{v[0-9]}}, [[VX:v[0-9]]], [[VT]], [[VR]]
define amdgpu_kernel void @test_f32_interp(float addrspace(1)* %out,
                             float addrspace(1)* %in1,
                             float addrspace(1)* %in2,
                             float addrspace(1)* %in3) {
  %x = load float, float addrspace(1)* %in1
  %y = load float, float addrspace(1)* %in2
  %t = load float, float addrspace(1)* %in3
  %t1 = fsub float 1.0, %t
  %tx = fmul float %x, %t
  %ty = fmul float %y, %t1
  %r = fadd float %tx, %ty
  store float %r, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_f64_interp:
; SI-NOFMA: v_add_f64 [[VT1:v\[[0-9]+:[0-9]+\]]], -[[VT:v\[[0-9]+:[0-9]+\]]], 1.0
; SI-NOFMA: v_mul_f64 [[VTY:v\[[0-9]+:[0-9]+\]]], [[VY:v\[[0-9]+:[0-9]+\]]], [[VT1]]
; SI-NOFMA: v_fma_f64 v{{\[[0-9]+:[0-9]+\]}}, [[VX:v\[[0-9]+:[0-9]+\]]], [[VT]], [[VTY]]
;
; SI-FMA: v_fma_f64 [[VR:v\[[0-9]+:[0-9]+\]]], -[[VT:v\[[0-9]+:[0-9]+\]]], [[VY:v\[[0-9]+:[0-9]+\]]], [[VY]]
; SI-FMA: v_fma_f64 v{{\[[0-9]+:[0-9]+\]}}, [[VX:v\[[0-9]+:[0-9]+\]]], [[VT]], [[VR]]
define amdgpu_kernel void @test_f64_interp(double addrspace(1)* %out,
                             double addrspace(1)* %in1,
                             double addrspace(1)* %in2,
                             double addrspace(1)* %in3) {
  %x = load double, double addrspace(1)* %in1
  %y = load double, double addrspace(1)* %in2
  %t = load double, double addrspace(1)* %in3
  %t1 = fsub double 1.0, %t
  %tx = fmul double %x, %t
  %ty = fmul double %y, %t1
  %r = fadd double %tx, %ty
  store double %r, double addrspace(1)* %out
  ret void
}

; Make sure negative constant cancels out fneg
; SI-LABEL: {{^}}fma_neg_2.0_neg_a_b_f32:
; SI: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; SI: {{buffer|flat|global}}_load_dword [[B:v[0-9]+]]
; SI-NOT: [[A]]
; SI-NOT: [[B]]
; SI: v_fma_f32 v{{[0-9]+}}, [[A]], 2.0, [[B]]
define amdgpu_kernel void @fma_neg_2.0_neg_a_b_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load volatile float, float addrspace(1)* %gep.0
  %r2 = load volatile float, float addrspace(1)* %gep.1

  %r1.fneg = fneg float %r1

  %r3 = tail call float @llvm.fma.f32(float -2.0, float %r1.fneg, float %r2)
  store float %r3, float addrspace(1)* %gep.out
  ret void
}

; SI-LABEL: {{^}}fma_2.0_neg_a_b_f32:
; SI: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; SI: {{buffer|flat|global}}_load_dword [[B:v[0-9]+]]
; SI-NOT: [[A]]
; SI-NOT: [[B]]
; SI: v_fma_f32 v{{[0-9]+}}, [[A]], -2.0, [[B]]
define amdgpu_kernel void @fma_2.0_neg_a_b_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load volatile float, float addrspace(1)* %gep.0
  %r2 = load volatile float, float addrspace(1)* %gep.1

  %r1.fneg = fneg float %r1

  %r3 = tail call float @llvm.fma.f32(float 2.0, float %r1.fneg, float %r2)
  store float %r3, float addrspace(1)* %gep.out
  ret void
}

; SI-LABEL: {{^}}fma_neg_b_c_v4f32:
; SI: v_fma_f32 v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}, -v{{[0-9]+}}
; SI: v_fma_f32 v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}, -v{{[0-9]+}}
; SI: v_fma_f32 v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}, -v{{[0-9]+}}
; SI: v_fma_f32 v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}, -v{{[0-9]+}}
define amdgpu_kernel void @fma_neg_b_c_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.0 = getelementptr <4 x float>, <4 x float> addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr <4 x float>, <4 x float> addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr <4 x float>, <4 x float> addrspace(1)* %gep.1, i32 2
  %gep.out = getelementptr <4 x float>, <4 x float> addrspace(1)* %out, i32 %tid

  %tmp0 = load <4 x float>, <4 x float> addrspace(1)* %gep.0
  %tmp1 = load <4 x float>, <4 x float> addrspace(1)* %gep.1
  %tmp2 = load <4 x float>, <4 x float> addrspace(1)* %gep.2

  %fneg0 = fneg fast <4 x float> %tmp0
  %fneg1 = fneg fast <4 x float> %tmp1
  %fma0 = tail call fast <4 x float> @llvm.fma.v4f32(<4 x float> %tmp2, <4 x float> %fneg0, <4 x float> %fneg1)

  store <4 x float> %fma0, <4 x float> addrspace(1)* %gep.out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { nounwind "no-signed-zeros-fp-math"="true" }

; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs -fp-contract=fast < %s | FileCheck -check-prefix=SI-FASTFMAF -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs -fp-contract=fast < %s | FileCheck -check-prefix=SI-SLOWFMAF -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.r600.read.tidig.x() #0
declare double @llvm.fabs.f64(double) #0
declare double @llvm.fma.f64(double, double, double) #0
declare float @llvm.fma.f32(float, float, float) #0

; (fadd (fmul x, y), z) -> (fma x, y, z)
; FUNC-LABEL: {{^}}combine_to_fma_f64_0:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[C]]
; SI: buffer_store_dwordx2 [[RESULT]]
define void @combine_to_fma_f64_0(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %a = load double addrspace(1)* %gep.0
  %b = load double addrspace(1)* %gep.1
  %c = load double addrspace(1)* %gep.2

  %mul = fmul double %a, %b
  %fma = fadd double %mul, %c
  store double %fma, double addrspace(1)* %gep.out
  ret void
}

; (fadd (fmul x, y), z) -> (fma x, y, z)
; FUNC-LABEL: {{^}}combine_to_fma_f64_0_2use:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI-DAG: buffer_load_dwordx2 [[D:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24{{$}}
; SI-DAG: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[C]]
; SI-DAG: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[D]]
; SI-DAG: buffer_store_dwordx2 [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dwordx2 [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI: s_endpgm
define void @combine_to_fma_f64_0_2use(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr double, double addrspace(1)* %gep.out.0, i32 1

  %a = load double addrspace(1)* %gep.0
  %b = load double addrspace(1)* %gep.1
  %c = load double addrspace(1)* %gep.2
  %d = load double addrspace(1)* %gep.3

  %mul = fmul double %a, %b
  %fma0 = fadd double %mul, %c
  %fma1 = fadd double %mul, %d
  store double %fma0, double addrspace(1)* %gep.out.0
  store double %fma1, double addrspace(1)* %gep.out.1
  ret void
}

; (fadd x, (fmul y, z)) -> (fma y, z, x)
; FUNC-LABEL: {{^}}combine_to_fma_f64_1:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[C]]
; SI: buffer_store_dwordx2 [[RESULT]]
define void @combine_to_fma_f64_1(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %a = load double addrspace(1)* %gep.0
  %b = load double addrspace(1)* %gep.1
  %c = load double addrspace(1)* %gep.2

  %mul = fmul double %a, %b
  %fma = fadd double %c, %mul
  store double %fma, double addrspace(1)* %gep.out
  ret void
}

; (fsub (fmul x, y), z) -> (fma x, y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_fma_fsub_0_f64:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], -[[C]]
; SI: buffer_store_dwordx2 [[RESULT]]
define void @combine_to_fma_fsub_0_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %a = load double addrspace(1)* %gep.0
  %b = load double addrspace(1)* %gep.1
  %c = load double addrspace(1)* %gep.2

  %mul = fmul double %a, %b
  %fma = fsub double %mul, %c
  store double %fma, double addrspace(1)* %gep.out
  ret void
}

; (fsub (fmul x, y), z) -> (fma x, y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_fma_fsub_f64_0_2use:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI-DAG: buffer_load_dwordx2 [[D:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24{{$}}
; SI-DAG: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], -[[C]]
; SI-DAG: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], -[[D]]
; SI-DAG: buffer_store_dwordx2 [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dwordx2 [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI: s_endpgm
define void @combine_to_fma_fsub_f64_0_2use(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr double, double addrspace(1)* %gep.out.0, i32 1

  %a = load double addrspace(1)* %gep.0
  %b = load double addrspace(1)* %gep.1
  %c = load double addrspace(1)* %gep.2
  %d = load double addrspace(1)* %gep.3

  %mul = fmul double %a, %b
  %fma0 = fsub double %mul, %c
  %fma1 = fsub double %mul, %d
  store double %fma0, double addrspace(1)* %gep.out.0
  store double %fma1, double addrspace(1)* %gep.out.1
  ret void
}

; (fsub x, (fmul y, z)) -> (fma (fneg y), z, x)
; FUNC-LABEL: {{^}}combine_to_fma_fsub_1_f64:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], [[C]]
; SI: buffer_store_dwordx2 [[RESULT]]
define void @combine_to_fma_fsub_1_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %a = load double addrspace(1)* %gep.0
  %b = load double addrspace(1)* %gep.1
  %c = load double addrspace(1)* %gep.2

  %mul = fmul double %a, %b
  %fma = fsub double %c, %mul
  store double %fma, double addrspace(1)* %gep.out
  ret void
}

; (fsub x, (fmul y, z)) -> (fma (fneg y), z, x)
; FUNC-LABEL: {{^}}combine_to_fma_fsub_1_f64_2use:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI-DAG: buffer_load_dwordx2 [[D:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24{{$}}
; SI-DAG: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], [[C]]
; SI-DAG: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], [[D]]
; SI-DAG: buffer_store_dwordx2 [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dwordx2 [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI: s_endpgm
define void @combine_to_fma_fsub_1_f64_2use(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr double, double addrspace(1)* %gep.out.0, i32 1

  %a = load double addrspace(1)* %gep.0
  %b = load double addrspace(1)* %gep.1
  %c = load double addrspace(1)* %gep.2
  %d = load double addrspace(1)* %gep.3

  %mul = fmul double %a, %b
  %fma0 = fsub double %c, %mul
  %fma1 = fsub double %d, %mul
  store double %fma0, double addrspace(1)* %gep.out.0
  store double %fma1, double addrspace(1)* %gep.out.1
  ret void
}

; (fsub (fneg (fmul x, y)), z) -> (fma (fneg x), y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_fma_fsub_2_f64:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], -[[C]]
; SI: buffer_store_dwordx2 [[RESULT]]
define void @combine_to_fma_fsub_2_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %a = load double addrspace(1)* %gep.0
  %b = load double addrspace(1)* %gep.1
  %c = load double addrspace(1)* %gep.2

  %mul = fmul double %a, %b
  %mul.neg = fsub double -0.0, %mul
  %fma = fsub double %mul.neg, %c

  store double %fma, double addrspace(1)* %gep.out
  ret void
}

; (fsub (fneg (fmul x, y)), z) -> (fma (fneg x), y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_fma_fsub_2_f64_2uses_neg:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI-DAG: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], -[[C]]
; SI-DAG: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], -[[D]]
; SI-DAG: buffer_store_dwordx2 [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dwordx2 [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI: s_endpgm
define void @combine_to_fma_fsub_2_f64_2uses_neg(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr double, double addrspace(1)* %gep.out.0, i32 1

  %a = load double addrspace(1)* %gep.0
  %b = load double addrspace(1)* %gep.1
  %c = load double addrspace(1)* %gep.2
  %d = load double addrspace(1)* %gep.3

  %mul = fmul double %a, %b
  %mul.neg = fsub double -0.0, %mul
  %fma0 = fsub double %mul.neg, %c
  %fma1 = fsub double %mul.neg, %d

  store double %fma0, double addrspace(1)* %gep.out.0
  store double %fma1, double addrspace(1)* %gep.out.1
  ret void
}

; (fsub (fneg (fmul x, y)), z) -> (fma (fneg x), y, (fneg z))
; FUNC-LABEL: {{^}}combine_to_fma_fsub_2_f64_2uses_mul:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI-DAG: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], -[[A]], [[B]], -[[C]]
; SI-DAG: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], -[[D]]
; SI-DAG: buffer_store_dwordx2 [[RESULT0]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_store_dwordx2 [[RESULT1]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI: s_endpgm
define void @combine_to_fma_fsub_2_f64_2uses_mul(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.out.0 = getelementptr double, double addrspace(1)* %out, i32 %tid
  %gep.out.1 = getelementptr double, double addrspace(1)* %gep.out.0, i32 1

  %a = load double addrspace(1)* %gep.0
  %b = load double addrspace(1)* %gep.1
  %c = load double addrspace(1)* %gep.2
  %d = load double addrspace(1)* %gep.3

  %mul = fmul double %a, %b
  %mul.neg = fsub double -0.0, %mul
  %fma0 = fsub double %mul.neg, %c
  %fma1 = fsub double %mul, %d

  store double %fma0, double addrspace(1)* %gep.out.0
  store double %fma1, double addrspace(1)* %gep.out.1
  ret void
}

; fold (fsub (fma x, y, (fmul u, v)), z) -> (fma x, y (fma u, v, (fneg z)))

; FUNC-LABEL: {{^}}aggressive_combine_to_fma_fsub_0_f64:
; SI-DAG: buffer_load_dwordx2 [[X:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[Y:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[Z:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI-DAG: buffer_load_dwordx2 [[U:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24{{$}}
; SI-DAG: buffer_load_dwordx2 [[V:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:32{{$}}
; SI: v_fma_f64 [[FMA0:v\[[0-9]+:[0-9]+\]]], [[U]], [[V]], -[[Z]]
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[X]], [[Y]], [[FMA0]]
; SI: buffer_store_dwordx2 [[RESULT]]
define void @aggressive_combine_to_fma_fsub_0_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.4 = getelementptr double, double addrspace(1)* %gep.0, i32 4
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %x = load double addrspace(1)* %gep.0
  %y = load double addrspace(1)* %gep.1
  %z = load double addrspace(1)* %gep.2
  %u = load double addrspace(1)* %gep.3
  %v = load double addrspace(1)* %gep.4

  %tmp0 = fmul double %u, %v
  %tmp1 = call double @llvm.fma.f64(double %x, double %y, double %tmp0) #0
  %tmp2 = fsub double %tmp1, %z

  store double %tmp2, double addrspace(1)* %gep.out
  ret void
}

; fold (fsub x, (fma y, z, (fmul u, v)))
;   -> (fma (fneg y), z, (fma (fneg u), v, x))

; FUNC-LABEL: {{^}}aggressive_combine_to_fma_fsub_1_f64:
; SI-DAG: buffer_load_dwordx2 [[X:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dwordx2 [[Y:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}
; SI-DAG: buffer_load_dwordx2 [[Z:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; SI-DAG: buffer_load_dwordx2 [[U:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24{{$}}
; SI-DAG: buffer_load_dwordx2 [[V:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:32{{$}}
; SI: v_fma_f64 [[FMA0:v\[[0-9]+:[0-9]+\]]], -[[U]], [[V]], [[X]]
; SI: v_fma_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], -[[Y]], [[Z]], [[FMA0]]
; SI: buffer_store_dwordx2 [[RESULT]]
define void @aggressive_combine_to_fma_fsub_1_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr double, double addrspace(1)* %gep.0, i32 2
  %gep.3 = getelementptr double, double addrspace(1)* %gep.0, i32 3
  %gep.4 = getelementptr double, double addrspace(1)* %gep.0, i32 4
  %gep.out = getelementptr double, double addrspace(1)* %out, i32 %tid

  %x = load double addrspace(1)* %gep.0
  %y = load double addrspace(1)* %gep.1
  %z = load double addrspace(1)* %gep.2
  %u = load double addrspace(1)* %gep.3
  %v = load double addrspace(1)* %gep.4

  %tmp0 = fmul double %u, %v
  %tmp1 = call double @llvm.fma.f64(double %y, double %z, double %tmp0) #0
  %tmp2 = fsub double %x, %tmp1

  store double %tmp2, double addrspace(1)* %gep.out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

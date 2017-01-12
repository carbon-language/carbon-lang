; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s

; GCN-LABEL: {{^}}v_fneg_add_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_sub_f32_e64 [[RESULT:v[0-9]+]], -[[A]], [[B]]
; GCN-NEXT: buffer_store_dword [[RESULT]]
define void @v_fneg_add_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %add = fadd float %a, %b
  %fneg = fsub float -0.000000e+00, %add
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_store_use_add_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_add_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG_ADD:v[0-9]+]], 0x80000000, [[ADD]]
; GCN-NEXT: buffer_store_dword [[NEG_ADD]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define void @v_fneg_add_store_use_add_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %add = fadd float %a, %b
  %fneg = fsub float -0.000000e+00, %add
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %add, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_multi_use_add_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_add_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG_ADD:v[0-9]+]], 0x80000000, [[ADD]]
; GCN: v_mul_f32_e32 [[MUL:v[0-9]+]], 4.0, [[ADD]]
; GCN-NEXT: buffer_store_dword [[NEG_ADD]]
; GCN-NEXT: buffer_store_dword [[MUL]]
define void @v_fneg_add_multi_use_add_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %add = fadd float %a, %b
  %fneg = fsub float -0.000000e+00, %add
  %use1 = fmul float %add, 4.0
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_subrev_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define void @v_fneg_add_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %add = fadd float %fneg.a, %b
  %fneg = fsub float -0.000000e+00, %add
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_x_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_subrev_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define void @v_fneg_add_x_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.b = fsub float -0.000000e+00, %b
  %add = fadd float %a, %fneg.b
  %fneg = fsub float -0.000000e+00, %add
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_fneg_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_add_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define void @v_fneg_add_fneg_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %fneg.b = fsub float -0.000000e+00, %b
  %add = fadd float %fneg.a, %fneg.b
  %fneg = fsub float -0.000000e+00, %add
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_store_use_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_xor_b32_e32 [[NEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN-DAG: v_subrev_f32_e32 [[NEG_ADD:v[0-9]+]], [[B]], [[A]]
; GCN-NEXT: buffer_store_dword [[NEG_ADD]]
; GCN-NEXT: buffer_store_dword [[NEG_A]]
define void @v_fneg_add_store_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %add = fadd float %fneg.a, %b
  %fneg = fsub float -0.000000e+00, %add
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %fneg.a, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_multi_use_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_subrev_f32_e32 [[NEG_ADD:v[0-9]+]], [[B]], [[A]]
; GCN-DAG: v_mul_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN-NEXT: buffer_store_dword [[NEG_ADD]]
; GCN-NEXT: buffer_store_dword [[MUL]]
define void @v_fneg_add_multi_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float %c) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %add = fadd float %fneg.a, %b
  %fneg = fsub float -0.000000e+00, %add
  %use1 = fmul float %fneg.a, %c
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; --------------------------------------------------------------------------------
; fmul tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_mul_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e64 [[RESULT:v[0-9]+]], [[A]], -[[B]]
; GCN-NEXT: buffer_store_dword [[RESULT]]
define void @v_fneg_mul_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = fmul float %a, %b
  %fneg = fsub float -0.000000e+00, %mul
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_store_use_mul_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG_MUL:v[0-9]+]], 0x80000000, [[ADD]]
; GCN-NEXT: buffer_store_dword [[NEG_MUL]]
; GCN: buffer_store_dword [[ADD]]
define void @v_fneg_mul_store_use_mul_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = fmul float %a, %b
  %fneg = fsub float -0.000000e+00, %mul
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %mul, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_multi_use_mul_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG_MUL:v[0-9]+]], 0x80000000, [[ADD]]
; GCN: v_mul_f32_e32 [[MUL:v[0-9]+]], 4.0, [[ADD]]
; GCN-NEXT: buffer_store_dword [[NEG_MUL]]
; GCN: buffer_store_dword [[MUL]]
define void @v_fneg_mul_multi_use_mul_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = fmul float %a, %b
  %fneg = fsub float -0.000000e+00, %mul
  %use1 = fmul float %mul, 4.0
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define void @v_fneg_mul_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %mul = fmul float %fneg.a, %b
  %fneg = fsub float -0.000000e+00, %mul
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_x_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define void @v_fneg_mul_x_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.b = fsub float -0.000000e+00, %b
  %mul = fmul float %a, %fneg.b
  %fneg = fsub float -0.000000e+00, %mul
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_fneg_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e64 [[ADD:v[0-9]+]], [[A]], -[[B]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define void @v_fneg_mul_fneg_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %fneg.b = fsub float -0.000000e+00, %b
  %mul = fmul float %fneg.a, %fneg.b
  %fneg = fsub float -0.000000e+00, %mul
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_store_use_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_xor_b32_e32 [[NEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN-DAG: v_mul_f32_e32 [[NEG_MUL:v[0-9]+]], [[B]], [[A]]
; GCN-NEXT: buffer_store_dword [[NEG_MUL]]
; GCN: buffer_store_dword [[NEG_A]]
define void @v_fneg_mul_store_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %mul = fmul float %fneg.a, %b
  %fneg = fsub float -0.000000e+00, %mul
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %fneg.a, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_multi_use_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[NEG_MUL:v[0-9]+]], [[B]], [[A]]
; GCN-DAG: v_mul_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN-NEXT: buffer_store_dword [[NEG_MUL]]
; GCN: buffer_store_dword [[MUL]]
define void @v_fneg_mul_multi_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float %c) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %mul = fmul float %fneg.a, %b
  %fneg = fsub float -0.000000e+00, %mul
  %use1 = fmul float %fneg.a, %c
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.r600.read.tidig.x() #0
declare float @llvm.fabs.f32(float) #0

; FUNC-LABEL: @mad_sub_f32
; SI: BUFFER_LOAD_DWORD [[REGA:v[0-9]+]]
; SI: BUFFER_LOAD_DWORD [[REGB:v[0-9]+]]
; SI: BUFFER_LOAD_DWORD [[REGC:v[0-9]+]]
; SI: V_MAD_F32 [[RESULT:v[0-9]+]], [[REGA]], [[REGB]], -[[REGC]]
; SI: BUFFER_STORE_DWORD [[RESULT]]
define void @mad_sub_f32(float addrspace(1)* noalias nocapture %out, float addrspace(1)* noalias nocapture readonly %ptr) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr float addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr float addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr float addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr float addrspace(1)* %out, i64 %tid.ext
  %a = load float addrspace(1)* %gep0, align 4
  %b = load float addrspace(1)* %gep1, align 4
  %c = load float addrspace(1)* %gep2, align 4
  %mul = fmul float %a, %b
  %sub = fsub float %mul, %c
  store float %sub, float addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @mad_sub_inv_f32
; SI: BUFFER_LOAD_DWORD [[REGA:v[0-9]+]]
; SI: BUFFER_LOAD_DWORD [[REGB:v[0-9]+]]
; SI: BUFFER_LOAD_DWORD [[REGC:v[0-9]+]]
; SI: V_MAD_F32 [[RESULT:v[0-9]+]], -[[REGA]], [[REGB]], [[REGC]]
; SI: BUFFER_STORE_DWORD [[RESULT]]
define void @mad_sub_inv_f32(float addrspace(1)* noalias nocapture %out, float addrspace(1)* noalias nocapture readonly %ptr) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr float addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr float addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr float addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr float addrspace(1)* %out, i64 %tid.ext
  %a = load float addrspace(1)* %gep0, align 4
  %b = load float addrspace(1)* %gep1, align 4
  %c = load float addrspace(1)* %gep2, align 4
  %mul = fmul float %a, %b
  %sub = fsub float %c, %mul
  store float %sub, float addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @mad_sub_f64
; SI: V_MUL_F64
; SI: V_ADD_F64
define void @mad_sub_f64(double addrspace(1)* noalias nocapture %out, double addrspace(1)* noalias nocapture readonly %ptr) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr double addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr double addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr double addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr double addrspace(1)* %out, i64 %tid.ext
  %a = load double addrspace(1)* %gep0, align 8
  %b = load double addrspace(1)* %gep1, align 8
  %c = load double addrspace(1)* %gep2, align 8
  %mul = fmul double %a, %b
  %sub = fsub double %mul, %c
  store double %sub, double addrspace(1)* %outgep, align 8
  ret void
}

; FUNC-LABEL: @mad_sub_fabs_f32
; SI: BUFFER_LOAD_DWORD [[REGA:v[0-9]+]]
; SI: BUFFER_LOAD_DWORD [[REGB:v[0-9]+]]
; SI: BUFFER_LOAD_DWORD [[REGC:v[0-9]+]]
; SI: V_MAD_F32 [[RESULT:v[0-9]+]], [[REGA]], [[REGB]], -|[[REGC]]|
; SI: BUFFER_STORE_DWORD [[RESULT]]
define void @mad_sub_fabs_f32(float addrspace(1)* noalias nocapture %out, float addrspace(1)* noalias nocapture readonly %ptr) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr float addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr float addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr float addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr float addrspace(1)* %out, i64 %tid.ext
  %a = load float addrspace(1)* %gep0, align 4
  %b = load float addrspace(1)* %gep1, align 4
  %c = load float addrspace(1)* %gep2, align 4
  %c.abs = call float @llvm.fabs.f32(float %c) #0
  %mul = fmul float %a, %b
  %sub = fsub float %mul, %c.abs
  store float %sub, float addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @mad_sub_fabs_inv_f32
; SI: BUFFER_LOAD_DWORD [[REGA:v[0-9]+]]
; SI: BUFFER_LOAD_DWORD [[REGB:v[0-9]+]]
; SI: BUFFER_LOAD_DWORD [[REGC:v[0-9]+]]
; SI: V_MAD_F32 [[RESULT:v[0-9]+]], -[[REGA]], [[REGB]], |[[REGC]]|
; SI: BUFFER_STORE_DWORD [[RESULT]]
define void @mad_sub_fabs_inv_f32(float addrspace(1)* noalias nocapture %out, float addrspace(1)* noalias nocapture readonly %ptr) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr float addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr float addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr float addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr float addrspace(1)* %out, i64 %tid.ext
  %a = load float addrspace(1)* %gep0, align 4
  %b = load float addrspace(1)* %gep1, align 4
  %c = load float addrspace(1)* %gep2, align 4
  %c.abs = call float @llvm.fabs.f32(float %c) #0
  %mul = fmul float %a, %b
  %sub = fsub float %c.abs, %mul
  store float %sub, float addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @neg_neg_mad_f32
; SI: V_MAD_F32 {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
define void @neg_neg_mad_f32(float addrspace(1)* noalias nocapture %out, float addrspace(1)* noalias nocapture readonly %ptr) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr float addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr float addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr float addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr float addrspace(1)* %out, i64 %tid.ext
  %a = load float addrspace(1)* %gep0, align 4
  %b = load float addrspace(1)* %gep1, align 4
  %c = load float addrspace(1)* %gep2, align 4
  %nega = fsub float -0.000000e+00, %a
  %negb = fsub float -0.000000e+00, %b
  %mul = fmul float %nega, %negb
  %sub = fadd float %mul, %c
  store float %sub, float addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @mad_fabs_sub_f32
; SI: BUFFER_LOAD_DWORD [[REGA:v[0-9]+]]
; SI: BUFFER_LOAD_DWORD [[REGB:v[0-9]+]]
; SI: BUFFER_LOAD_DWORD [[REGC:v[0-9]+]]
; SI: V_MAD_F32 [[RESULT:v[0-9]+]], [[REGA]], |[[REGB]]|, -[[REGC]]
; SI: BUFFER_STORE_DWORD [[RESULT]]
define void @mad_fabs_sub_f32(float addrspace(1)* noalias nocapture %out, float addrspace(1)* noalias nocapture readonly %ptr) #1 {
  %tid = tail call i32 @llvm.r600.read.tidig.x() #0
  %tid.ext = sext i32 %tid to i64
  %gep0 = getelementptr float addrspace(1)* %ptr, i64 %tid.ext
  %add1 = add i64 %tid.ext, 1
  %gep1 = getelementptr float addrspace(1)* %ptr, i64 %add1
  %add2 = add i64 %tid.ext, 2
  %gep2 = getelementptr float addrspace(1)* %ptr, i64 %add2
  %outgep = getelementptr float addrspace(1)* %out, i64 %tid.ext
  %a = load float addrspace(1)* %gep0, align 4
  %b = load float addrspace(1)* %gep1, align 4
  %c = load float addrspace(1)* %gep2, align 4
  %b.abs = call float @llvm.fabs.f32(float %b) #0
  %mul = fmul float %a, %b.abs
  %sub = fsub float %mul, %c
  store float %sub, float addrspace(1)* %outgep, align 4
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

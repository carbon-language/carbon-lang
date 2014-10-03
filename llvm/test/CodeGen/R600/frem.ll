; RUN: llc -march=r600 -mcpu=SI -enable-misched < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}frem_f32:
; SI-DAG: BUFFER_LOAD_DWORD [[X:v[0-9]+]], {{.*$}}
; SI-DAG: BUFFER_LOAD_DWORD [[Y:v[0-9]+]], {{.*}} offset:0x10
; SI-DAG: V_CMP
; SI-DAG: V_MUL_F32
; SI: V_RCP_F32_e32
; SI: V_MUL_F32_e32
; SI: V_MUL_F32_e32
; SI: V_TRUNC_F32_e32
; SI: V_MAD_F32
; SI: S_ENDPGM
define void @frem_f32(float addrspace(1)* %out, float addrspace(1)* %in1,
                      float addrspace(1)* %in2) #0 {
   %gep2 = getelementptr float addrspace(1)* %in2, i32 4
   %r0 = load float addrspace(1)* %in1, align 4
   %r1 = load float addrspace(1)* %gep2, align 4
   %r2 = frem float %r0, %r1
   store float %r2, float addrspace(1)* %out, align 4
   ret void
}

; FUNC-LABEL: {{^}}unsafe_frem_f32:
; SI: BUFFER_LOAD_DWORD [[Y:v[0-9]+]], {{.*}} offset:0x10
; SI: BUFFER_LOAD_DWORD [[X:v[0-9]+]], {{.*}}
; SI: V_RCP_F32_e32 [[INVY:v[0-9]+]], [[Y]]
; SI: V_MUL_F32_e32 [[DIV:v[0-9]+]], [[INVY]], [[X]]
; SI: V_TRUNC_F32_e32 [[TRUNC:v[0-9]+]], [[DIV]]
; SI: V_MAD_F32 [[RESULT:v[0-9]+]], -[[TRUNC]], [[Y]], [[X]]
; SI: BUFFER_STORE_DWORD [[RESULT]]
; SI: S_ENDPGM
define void @unsafe_frem_f32(float addrspace(1)* %out, float addrspace(1)* %in1,
                             float addrspace(1)* %in2) #1 {
   %gep2 = getelementptr float addrspace(1)* %in2, i32 4
   %r0 = load float addrspace(1)* %in1, align 4
   %r1 = load float addrspace(1)* %gep2, align 4
   %r2 = frem float %r0, %r1
   store float %r2, float addrspace(1)* %out, align 4
   ret void
}

; TODO: This should check something when f64 fdiv is implemented
; correctly

; FUNC-LABEL: {{^}}frem_f64:
; SI: S_ENDPGM
define void @frem_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                      double addrspace(1)* %in2) #0 {
   %r0 = load double addrspace(1)* %in1, align 8
   %r1 = load double addrspace(1)* %in2, align 8
   %r2 = frem double %r0, %r1
   store double %r2, double addrspace(1)* %out, align 8
   ret void
}

; FUNC-LABEL: {{^}}unsafe_frem_f64:
; SI: V_RCP_F64_e32
; SI: V_MUL_F64
; SI: V_BFE_U32
; SI: V_FMA_F64
; SI: S_ENDPGM
define void @unsafe_frem_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                             double addrspace(1)* %in2) #1 {
   %r0 = load double addrspace(1)* %in1, align 8
   %r1 = load double addrspace(1)* %in2, align 8
   %r2 = frem double %r0, %r1
   store double %r2, double addrspace(1)* %out, align 8
   ret void
}

define void @frem_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %in1,
                        <2 x float> addrspace(1)* %in2) #0 {
   %gep2 = getelementptr <2 x float> addrspace(1)* %in2, i32 4
   %r0 = load <2 x float> addrspace(1)* %in1, align 8
   %r1 = load <2 x float> addrspace(1)* %gep2, align 8
   %r2 = frem <2 x float> %r0, %r1
   store <2 x float> %r2, <2 x float> addrspace(1)* %out, align 8
   ret void
}

define void @frem_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in1,
                        <4 x float> addrspace(1)* %in2) #0 {
   %gep2 = getelementptr <4 x float> addrspace(1)* %in2, i32 4
   %r0 = load <4 x float> addrspace(1)* %in1, align 16
   %r1 = load <4 x float> addrspace(1)* %gep2, align 16
   %r2 = frem <4 x float> %r0, %r1
   store <4 x float> %r2, <4 x float> addrspace(1)* %out, align 16
   ret void
}

define void @frem_v2f64(<2 x double> addrspace(1)* %out, <2 x double> addrspace(1)* %in1,
                        <2 x double> addrspace(1)* %in2) #0 {
   %gep2 = getelementptr <2 x double> addrspace(1)* %in2, i32 4
   %r0 = load <2 x double> addrspace(1)* %in1, align 16
   %r1 = load <2 x double> addrspace(1)* %gep2, align 16
   %r2 = frem <2 x double> %r0, %r1
   store <2 x double> %r2, <2 x double> addrspace(1)* %out, align 16
   ret void
}

attributes #0 = { nounwind "unsafe-fp-math"="false" }
attributes #1 = { nounwind "unsafe-fp-math"="true" }

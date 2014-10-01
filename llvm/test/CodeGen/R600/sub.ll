;RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
;RUN: llc -march=r600 -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.r600.read.tidig.x() readnone

;FUNC-LABEL: {{^}}test2:
;EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI: V_SUB_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: V_SUB_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @test2(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32> addrspace(1) * %in
  %b = load <2 x i32> addrspace(1) * %b_ptr
  %result = sub <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

;FUNC-LABEL: {{^}}test4:
;EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI: V_SUB_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: V_SUB_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: V_SUB_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: V_SUB_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @test4(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32> addrspace(1) * %in
  %b = load <4 x i32> addrspace(1) * %b_ptr
  %result = sub <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_sub_i64:
; SI: S_SUB_U32
; SI: S_SUBB_U32

; EG-DAG: SETGE_UINT
; EG-DAG: CNDE_INT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
define void @s_sub_i64(i64 addrspace(1)* noalias %out, i64 %a, i64 %b) nounwind {
  %result = sub i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_sub_i64:
; SI: V_SUB_I32_e32
; SI: V_SUBB_U32_e32

; EG-DAG: SETGE_UINT
; EG-DAG: CNDE_INT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
define void @v_sub_i64(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %inA, i64 addrspace(1)* noalias %inB) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() readnone
  %a_ptr = getelementptr i64 addrspace(1)* %inA, i32 %tid
  %b_ptr = getelementptr i64 addrspace(1)* %inB, i32 %tid
  %a = load i64 addrspace(1)* %a_ptr
  %b = load i64 addrspace(1)* %b_ptr
  %result = sub i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

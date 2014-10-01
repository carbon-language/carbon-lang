; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i32 @llvm.cttz.i32(i32, i1) nounwind readnone
declare <2 x i32> @llvm.cttz.v2i32(<2 x i32>, i1) nounwind readnone
declare <4 x i32> @llvm.cttz.v4i32(<4 x i32>, i1) nounwind readnone

; FUNC-LABEL: {{^}}s_cttz_zero_undef_i32:
; SI: S_LOAD_DWORD [[VAL:s[0-9]+]],
; SI: S_FF1_I32_B32 [[SRESULT:s[0-9]+]], [[VAL]]
; SI: V_MOV_B32_e32 [[VRESULT:v[0-9]+]], [[SRESULT]]
; SI: BUFFER_STORE_DWORD [[VRESULT]],
; SI: S_ENDPGM
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+\.[XYZW]]]
; EG: FFBL_INT {{\*? *}}[[RESULT]]
define void @s_cttz_zero_undef_i32(i32 addrspace(1)* noalias %out, i32 %val) nounwind {
  %cttz = call i32 @llvm.cttz.i32(i32 %val, i1 true) nounwind readnone
  store i32 %cttz, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_cttz_zero_undef_i32:
; SI: BUFFER_LOAD_DWORD [[VAL:v[0-9]+]],
; SI: V_FFBL_B32_e32 [[RESULT:v[0-9]+]], [[VAL]]
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+\.[XYZW]]]
; EG: FFBL_INT {{\*? *}}[[RESULT]]
define void @v_cttz_zero_undef_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) nounwind {
  %val = load i32 addrspace(1)* %valptr, align 4
  %cttz = call i32 @llvm.cttz.i32(i32 %val, i1 true) nounwind readnone
  store i32 %cttz, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_cttz_zero_undef_v2i32:
; SI: BUFFER_LOAD_DWORDX2
; SI: V_FFBL_B32_e32
; SI: V_FFBL_B32_e32
; SI: BUFFER_STORE_DWORDX2
; SI: S_ENDPGM
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+]]{{\.[XYZW]}}
; EG: FFBL_INT {{\*? *}}[[RESULT]]
; EG: FFBL_INT {{\*? *}}[[RESULT]]
define void @v_cttz_zero_undef_v2i32(<2 x i32> addrspace(1)* noalias %out, <2 x i32> addrspace(1)* noalias %valptr) nounwind {
  %val = load <2 x i32> addrspace(1)* %valptr, align 8
  %cttz = call <2 x i32> @llvm.cttz.v2i32(<2 x i32> %val, i1 true) nounwind readnone
  store <2 x i32> %cttz, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_cttz_zero_undef_v4i32:
; SI: BUFFER_LOAD_DWORDX4
; SI: V_FFBL_B32_e32
; SI: V_FFBL_B32_e32
; SI: V_FFBL_B32_e32
; SI: V_FFBL_B32_e32
; SI: BUFFER_STORE_DWORDX4
; SI: S_ENDPGM
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+]]{{\.[XYZW]}}
; EG: FFBL_INT {{\*? *}}[[RESULT]]
; EG: FFBL_INT {{\*? *}}[[RESULT]]
; EG: FFBL_INT {{\*? *}}[[RESULT]]
; EG: FFBL_INT {{\*? *}}[[RESULT]]
define void @v_cttz_zero_undef_v4i32(<4 x i32> addrspace(1)* noalias %out, <4 x i32> addrspace(1)* noalias %valptr) nounwind {
  %val = load <4 x i32> addrspace(1)* %valptr, align 16
  %cttz = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %val, i1 true) nounwind readnone
  store <4 x i32> %cttz, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

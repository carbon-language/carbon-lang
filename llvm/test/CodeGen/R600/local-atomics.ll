; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=CI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}lds_atomic_xchg_ret_i32:
; EG: LDS_WRXCHG_RET *
; SI: S_LOAD_DWORD [[SPTR:s[0-9]+]],
; SI: V_MOV_B32_e32 [[DATA:v[0-9]+]], 4
; SI: V_MOV_B32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; SI: DS_WRXCHG_RTN_B32 [[RESULT:v[0-9]+]], [[VPTR]], [[DATA]] [M0]
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM
define void @lds_atomic_xchg_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw xchg i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xchg_ret_i32_offset:
; EG: LDS_WRXCHG_RET *
; SI: DS_WRXCHG_RTN_B32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_xchg_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw xchg i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; XXX - Is it really necessary to load 4 into VGPR?
; FUNC-LABEL: {{^}}lds_atomic_add_ret_i32:
; EG: LDS_ADD_RET *
; SI: S_LOAD_DWORD [[SPTR:s[0-9]+]],
; SI: V_MOV_B32_e32 [[DATA:v[0-9]+]], 4
; SI: V_MOV_B32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; SI: DS_ADD_RTN_U32 [[RESULT:v[0-9]+]], [[VPTR]], [[DATA]] [M0]
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM
define void @lds_atomic_add_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw add i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add_ret_i32_offset:
; EG: LDS_ADD_RET *
; SI: DS_ADD_RTN_U32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_add_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add_ret_i32_bad_si_offset:
; EG: LDS_ADD_RET *
; SI: DS_ADD_RTN_U32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} [M0]
; CI: DS_ADD_RTN_U32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_add_ret_i32_bad_si_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr, i32 %a, i32 %b) nounwind {
  %sub = sub i32 %a, %b
  %add = add i32 %sub, 4
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 %add
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_inc_ret_i32:
; EG: LDS_ADD_RET *
; SI: S_MOV_B32 [[SNEGONE:s[0-9]+]], -1
; SI: V_MOV_B32_e32 [[NEGONE:v[0-9]+]], [[SNEGONE]]
; SI: DS_INC_RTN_U32 v{{[0-9]+}}, v{{[0-9]+}}, [[NEGONE]] [M0]
; SI: S_ENDPGM
define void @lds_atomic_inc_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw add i32 addrspace(3)* %ptr, i32 1 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_inc_ret_i32_offset:
; EG: LDS_ADD_RET *
; SI: S_MOV_B32 [[SNEGONE:s[0-9]+]], -1
; SI: V_MOV_B32_e32 [[NEGONE:v[0-9]+]], [[SNEGONE]]
; SI: DS_INC_RTN_U32 v{{[0-9]+}}, v{{[0-9]+}}, [[NEGONE]] offset:16
; SI: S_ENDPGM
define void @lds_atomic_inc_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 1 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_inc_ret_i32_bad_si_offset:
; EG: LDS_ADD_RET *
; SI: DS_INC_RTN_U32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} [M0]
; CI: DS_INC_RTN_U32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_inc_ret_i32_bad_si_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr, i32 %a, i32 %b) nounwind {
  %sub = sub i32 %a, %b
  %add = add i32 %sub, 4
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 %add
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 1 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_sub_ret_i32:
; EG: LDS_SUB_RET *
; SI: DS_SUB_RTN_U32
; SI: S_ENDPGM
define void @lds_atomic_sub_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw sub i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_sub_ret_i32_offset:
; EG: LDS_SUB_RET *
; SI: DS_SUB_RTN_U32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_sub_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw sub i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_dec_ret_i32:
; EG: LDS_SUB_RET *
; SI: S_MOV_B32 [[SNEGONE:s[0-9]+]], -1
; SI: V_MOV_B32_e32 [[NEGONE:v[0-9]+]], [[SNEGONE]]
; SI: DS_DEC_RTN_U32  v{{[0-9]+}}, v{{[0-9]+}}, [[NEGONE]] [M0]
; SI: S_ENDPGM
define void @lds_atomic_dec_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw sub i32 addrspace(3)* %ptr, i32 1 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_dec_ret_i32_offset:
; EG: LDS_SUB_RET *
; SI: S_MOV_B32 [[SNEGONE:s[0-9]+]], -1
; SI: V_MOV_B32_e32 [[NEGONE:v[0-9]+]], [[SNEGONE]]
; SI: DS_DEC_RTN_U32 v{{[0-9]+}}, v{{[0-9]+}}, [[NEGONE]] offset:16
; SI: S_ENDPGM
define void @lds_atomic_dec_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw sub i32 addrspace(3)* %gep, i32 1 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_and_ret_i32:
; EG: LDS_AND_RET *
; SI: DS_AND_RTN_B32
; SI: S_ENDPGM
define void @lds_atomic_and_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw and i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_and_ret_i32_offset:
; EG: LDS_AND_RET *
; SI: DS_AND_RTN_B32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_and_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw and i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_or_ret_i32:
; EG: LDS_OR_RET *
; SI: DS_OR_RTN_B32
; SI: S_ENDPGM
define void @lds_atomic_or_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw or i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_or_ret_i32_offset:
; EG: LDS_OR_RET *
; SI: DS_OR_RTN_B32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_or_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw or i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xor_ret_i32:
; EG: LDS_XOR_RET *
; SI: DS_XOR_RTN_B32
; SI: S_ENDPGM
define void @lds_atomic_xor_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw xor i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xor_ret_i32_offset:
; EG: LDS_XOR_RET *
; SI: DS_XOR_RTN_B32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_xor_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw xor i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FIXME: There is no atomic nand instr
; XFUNC-LABEL: {{^}}lds_atomic_nand_ret_i32:uction, so we somehow need to expand this.
; define void @lds_atomic_nand_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
;   %result = atomicrmw nand i32 addrspace(3)* %ptr, i32 4 seq_cst
;   store i32 %result, i32 addrspace(1)* %out, align 4
;   ret void
; }

; FUNC-LABEL: {{^}}lds_atomic_min_ret_i32:
; EG: LDS_MIN_INT_RET *
; SI: DS_MIN_RTN_I32
; SI: S_ENDPGM
define void @lds_atomic_min_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw min i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_min_ret_i32_offset:
; EG: LDS_MIN_INT_RET *
; SI: DS_MIN_RTN_I32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_min_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw min i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_max_ret_i32:
; EG: LDS_MAX_INT_RET *
; SI: DS_MAX_RTN_I32
; SI: S_ENDPGM
define void @lds_atomic_max_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw max i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_max_ret_i32_offset:
; EG: LDS_MAX_INT_RET *
; SI: DS_MAX_RTN_I32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_max_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw max i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umin_ret_i32:
; EG: LDS_MIN_UINT_RET *
; SI: DS_MIN_RTN_U32
; SI: S_ENDPGM
define void @lds_atomic_umin_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw umin i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umin_ret_i32_offset:
; EG: LDS_MIN_UINT_RET *
; SI: DS_MIN_RTN_U32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_umin_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw umin i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umax_ret_i32:
; EG: LDS_MAX_UINT_RET *
; SI: DS_MAX_RTN_U32
; SI: S_ENDPGM
define void @lds_atomic_umax_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw umax i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umax_ret_i32_offset:
; EG: LDS_MAX_UINT_RET *
; SI: DS_MAX_RTN_U32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_umax_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw umax i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xchg_noret_i32:
; SI: S_LOAD_DWORD [[SPTR:s[0-9]+]],
; SI: V_MOV_B32_e32 [[DATA:v[0-9]+]], 4
; SI: V_MOV_B32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; SI: DS_WRXCHG_RTN_B32 [[RESULT:v[0-9]+]], [[VPTR]], [[DATA]] [M0]
; SI: S_ENDPGM
define void @lds_atomic_xchg_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw xchg i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xchg_noret_i32_offset:
; SI: DS_WRXCHG_RTN_B32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_xchg_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw xchg i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; XXX - Is it really necessary to load 4 into VGPR?
; FUNC-LABEL: {{^}}lds_atomic_add_noret_i32:
; SI: S_LOAD_DWORD [[SPTR:s[0-9]+]],
; SI: V_MOV_B32_e32 [[DATA:v[0-9]+]], 4
; SI: V_MOV_B32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; SI: DS_ADD_U32 [[VPTR]], [[DATA]] [M0]
; SI: S_ENDPGM
define void @lds_atomic_add_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw add i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add_noret_i32_offset:
; SI: DS_ADD_U32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_add_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add_noret_i32_bad_si_offset
; SI: DS_ADD_U32 v{{[0-9]+}}, v{{[0-9]+}} [M0]
; CI: DS_ADD_U32 v{{[0-9]+}}, v{{[0-9]+}} offset:16 [M0]
; SI: S_ENDPGM
define void @lds_atomic_add_noret_i32_bad_si_offset(i32 addrspace(3)* %ptr, i32 %a, i32 %b) nounwind {
  %sub = sub i32 %a, %b
  %add = add i32 %sub, 4
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 %add
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_inc_noret_i32:
; SI: S_MOV_B32 [[SNEGONE:s[0-9]+]], -1
; SI: V_MOV_B32_e32 [[NEGONE:v[0-9]+]], [[SNEGONE]]
; SI: DS_INC_U32 v{{[0-9]+}}, [[NEGONE]] [M0]
; SI: S_ENDPGM
define void @lds_atomic_inc_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw add i32 addrspace(3)* %ptr, i32 1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_inc_noret_i32_offset:
; SI: S_MOV_B32 [[SNEGONE:s[0-9]+]], -1
; SI: V_MOV_B32_e32 [[NEGONE:v[0-9]+]], [[SNEGONE]]
; SI: DS_INC_U32 v{{[0-9]+}}, [[NEGONE]] offset:16
; SI: S_ENDPGM
define void @lds_atomic_inc_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_inc_noret_i32_bad_si_offset:
; SI: DS_INC_U32 v{{[0-9]+}}, v{{[0-9]+}}
; CI: DS_INC_U32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_inc_noret_i32_bad_si_offset(i32 addrspace(3)* %ptr, i32 %a, i32 %b) nounwind {
  %sub = sub i32 %a, %b
  %add = add i32 %sub, 4
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 %add
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_sub_noret_i32:
; SI: DS_SUB_U32
; SI: S_ENDPGM
define void @lds_atomic_sub_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw sub i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_sub_noret_i32_offset:
; SI: DS_SUB_U32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_sub_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw sub i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_dec_noret_i32:
; SI: S_MOV_B32 [[SNEGONE:s[0-9]+]], -1
; SI: V_MOV_B32_e32 [[NEGONE:v[0-9]+]], [[SNEGONE]]
; SI: DS_DEC_U32  v{{[0-9]+}}, [[NEGONE]]
; SI: S_ENDPGM
define void @lds_atomic_dec_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw sub i32 addrspace(3)* %ptr, i32 1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_dec_noret_i32_offset:
; SI: S_MOV_B32 [[SNEGONE:s[0-9]+]], -1
; SI: V_MOV_B32_e32 [[NEGONE:v[0-9]+]], [[SNEGONE]]
; SI: DS_DEC_U32 v{{[0-9]+}}, [[NEGONE]] offset:16
; SI: S_ENDPGM
define void @lds_atomic_dec_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw sub i32 addrspace(3)* %gep, i32 1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_and_noret_i32:
; SI: DS_AND_B32
; SI: S_ENDPGM
define void @lds_atomic_and_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw and i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_and_noret_i32_offset:
; SI: DS_AND_B32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_and_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw and i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_or_noret_i32:
; SI: DS_OR_B32
; SI: S_ENDPGM
define void @lds_atomic_or_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw or i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_or_noret_i32_offset:
; SI: DS_OR_B32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_or_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw or i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xor_noret_i32:
; SI: DS_XOR_B32
; SI: S_ENDPGM
define void @lds_atomic_xor_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw xor i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xor_noret_i32_offset:
; SI: DS_XOR_B32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_xor_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw xor i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FIXME: There is no atomic nand instr
; XFUNC-LABEL: {{^}}lds_atomic_nand_noret_i32:uction, so we somehow need to expand this.
; define void @lds_atomic_nand_noret_i32(i32 addrspace(3)* %ptr) nounwind {
;   %result = atomicrmw nand i32 addrspace(3)* %ptr, i32 4 seq_cst
;   ret void
; }

; FUNC-LABEL: {{^}}lds_atomic_min_noret_i32:
; SI: DS_MIN_I32
; SI: S_ENDPGM
define void @lds_atomic_min_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw min i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_min_noret_i32_offset:
; SI: DS_MIN_I32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_min_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw min i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_max_noret_i32:
; SI: DS_MAX_I32
; SI: S_ENDPGM
define void @lds_atomic_max_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw max i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_max_noret_i32_offset:
; SI: DS_MAX_I32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_max_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw max i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umin_noret_i32:
; SI: DS_MIN_U32
; SI: S_ENDPGM
define void @lds_atomic_umin_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw umin i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umin_noret_i32_offset:
; SI: DS_MIN_U32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_umin_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw umin i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umax_noret_i32:
; SI: DS_MAX_U32
; SI: S_ENDPGM
define void @lds_atomic_umax_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw umax i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umax_noret_i32_offset:
; SI: DS_MAX_U32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: S_ENDPGM
define void @lds_atomic_umax_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw umax i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

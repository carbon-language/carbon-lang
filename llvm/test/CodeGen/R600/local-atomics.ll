; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=CI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}lds_atomic_xchg_ret_i32:
; EG: LDS_WRXCHG_RET *
; SI: s_load_dword [[SPTR:s[0-9]+]],
; SI: v_mov_b32_e32 [[DATA:v[0-9]+]], 4
; SI: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; SI: ds_wrxchg_rtn_b32 [[RESULT:v[0-9]+]], [[VPTR]], [[DATA]] [M0]
; SI: buffer_store_dword [[RESULT]],
; SI: s_endpgm
define void @lds_atomic_xchg_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw xchg i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xchg_ret_i32_offset:
; EG: LDS_WRXCHG_RET *
; SI: ds_wrxchg_rtn_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_xchg_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw xchg i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; XXX - Is it really necessary to load 4 into VGPR?
; FUNC-LABEL: {{^}}lds_atomic_add_ret_i32:
; EG: LDS_ADD_RET *
; SI: s_load_dword [[SPTR:s[0-9]+]],
; SI: v_mov_b32_e32 [[DATA:v[0-9]+]], 4
; SI: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; SI: ds_add_rtn_u32 [[RESULT:v[0-9]+]], [[VPTR]], [[DATA]] [M0]
; SI: buffer_store_dword [[RESULT]],
; SI: s_endpgm
define void @lds_atomic_add_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw add i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add_ret_i32_offset:
; EG: LDS_ADD_RET *
; SI: ds_add_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_add_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add_ret_i32_bad_si_offset:
; EG: LDS_ADD_RET *
; SI: ds_add_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} [M0]
; CI: ds_add_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
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
; SI: v_mov_b32_e32 [[NEGONE:v[0-9]+]], -1
; SI: ds_inc_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, [[NEGONE]] [M0]
; SI: s_endpgm
define void @lds_atomic_inc_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw add i32 addrspace(3)* %ptr, i32 1 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_inc_ret_i32_offset:
; EG: LDS_ADD_RET *
; SI: v_mov_b32_e32 [[NEGONE:v[0-9]+]], -1
; SI: ds_inc_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, [[NEGONE]] offset:16
; SI: s_endpgm
define void @lds_atomic_inc_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 1 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_inc_ret_i32_bad_si_offset:
; EG: LDS_ADD_RET *
; SI: ds_inc_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} [M0]
; CI: ds_inc_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
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
; SI: ds_sub_rtn_u32
; SI: s_endpgm
define void @lds_atomic_sub_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw sub i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_sub_ret_i32_offset:
; EG: LDS_SUB_RET *
; SI: ds_sub_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_sub_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw sub i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_dec_ret_i32:
; EG: LDS_SUB_RET *
; SI: v_mov_b32_e32 [[NEGONE:v[0-9]+]], -1
; SI: ds_dec_rtn_u32  v{{[0-9]+}}, v{{[0-9]+}}, [[NEGONE]] [M0]
; SI: s_endpgm
define void @lds_atomic_dec_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw sub i32 addrspace(3)* %ptr, i32 1 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_dec_ret_i32_offset:
; EG: LDS_SUB_RET *
; SI: v_mov_b32_e32 [[NEGONE:v[0-9]+]], -1
; SI: ds_dec_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, [[NEGONE]] offset:16
; SI: s_endpgm
define void @lds_atomic_dec_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw sub i32 addrspace(3)* %gep, i32 1 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_and_ret_i32:
; EG: LDS_AND_RET *
; SI: ds_and_rtn_b32
; SI: s_endpgm
define void @lds_atomic_and_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw and i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_and_ret_i32_offset:
; EG: LDS_AND_RET *
; SI: ds_and_rtn_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_and_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw and i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_or_ret_i32:
; EG: LDS_OR_RET *
; SI: ds_or_rtn_b32
; SI: s_endpgm
define void @lds_atomic_or_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw or i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_or_ret_i32_offset:
; EG: LDS_OR_RET *
; SI: ds_or_rtn_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_or_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw or i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xor_ret_i32:
; EG: LDS_XOR_RET *
; SI: ds_xor_rtn_b32
; SI: s_endpgm
define void @lds_atomic_xor_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw xor i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xor_ret_i32_offset:
; EG: LDS_XOR_RET *
; SI: ds_xor_rtn_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
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
; SI: ds_min_rtn_i32
; SI: s_endpgm
define void @lds_atomic_min_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw min i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_min_ret_i32_offset:
; EG: LDS_MIN_INT_RET *
; SI: ds_min_rtn_i32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_min_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw min i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_max_ret_i32:
; EG: LDS_MAX_INT_RET *
; SI: ds_max_rtn_i32
; SI: s_endpgm
define void @lds_atomic_max_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw max i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_max_ret_i32_offset:
; EG: LDS_MAX_INT_RET *
; SI: ds_max_rtn_i32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_max_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw max i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umin_ret_i32:
; EG: LDS_MIN_UINT_RET *
; SI: ds_min_rtn_u32
; SI: s_endpgm
define void @lds_atomic_umin_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw umin i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umin_ret_i32_offset:
; EG: LDS_MIN_UINT_RET *
; SI: ds_min_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_umin_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw umin i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umax_ret_i32:
; EG: LDS_MAX_UINT_RET *
; SI: ds_max_rtn_u32
; SI: s_endpgm
define void @lds_atomic_umax_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw umax i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umax_ret_i32_offset:
; EG: LDS_MAX_UINT_RET *
; SI: ds_max_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_umax_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw umax i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xchg_noret_i32:
; SI: s_load_dword [[SPTR:s[0-9]+]],
; SI: v_mov_b32_e32 [[DATA:v[0-9]+]], 4
; SI: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; SI: ds_wrxchg_rtn_b32 [[RESULT:v[0-9]+]], [[VPTR]], [[DATA]] [M0]
; SI: s_endpgm
define void @lds_atomic_xchg_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw xchg i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xchg_noret_i32_offset:
; SI: ds_wrxchg_rtn_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_xchg_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw xchg i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; XXX - Is it really necessary to load 4 into VGPR?
; FUNC-LABEL: {{^}}lds_atomic_add_noret_i32:
; SI: s_load_dword [[SPTR:s[0-9]+]],
; SI: v_mov_b32_e32 [[DATA:v[0-9]+]], 4
; SI: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; SI: ds_add_u32 [[VPTR]], [[DATA]] [M0]
; SI: s_endpgm
define void @lds_atomic_add_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw add i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add_noret_i32_offset:
; SI: ds_add_u32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_add_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add_noret_i32_bad_si_offset
; SI: ds_add_u32 v{{[0-9]+}}, v{{[0-9]+}} [M0]
; CI: ds_add_u32 v{{[0-9]+}}, v{{[0-9]+}} offset:16 [M0]
; SI: s_endpgm
define void @lds_atomic_add_noret_i32_bad_si_offset(i32 addrspace(3)* %ptr, i32 %a, i32 %b) nounwind {
  %sub = sub i32 %a, %b
  %add = add i32 %sub, 4
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 %add
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_inc_noret_i32:
; SI: v_mov_b32_e32 [[NEGONE:v[0-9]+]], -1
; SI: ds_inc_u32 v{{[0-9]+}}, [[NEGONE]] [M0]
; SI: s_endpgm
define void @lds_atomic_inc_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw add i32 addrspace(3)* %ptr, i32 1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_inc_noret_i32_offset:
; SI: v_mov_b32_e32 [[NEGONE:v[0-9]+]], -1
; SI: ds_inc_u32 v{{[0-9]+}}, [[NEGONE]] offset:16
; SI: s_endpgm
define void @lds_atomic_inc_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_inc_noret_i32_bad_si_offset:
; SI: ds_inc_u32 v{{[0-9]+}}, v{{[0-9]+}}
; CI: ds_inc_u32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_inc_noret_i32_bad_si_offset(i32 addrspace(3)* %ptr, i32 %a, i32 %b) nounwind {
  %sub = sub i32 %a, %b
  %add = add i32 %sub, 4
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 %add
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_sub_noret_i32:
; SI: ds_sub_u32
; SI: s_endpgm
define void @lds_atomic_sub_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw sub i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_sub_noret_i32_offset:
; SI: ds_sub_u32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_sub_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw sub i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_dec_noret_i32:
; SI: v_mov_b32_e32 [[NEGONE:v[0-9]+]], -1
; SI: ds_dec_u32  v{{[0-9]+}}, [[NEGONE]]
; SI: s_endpgm
define void @lds_atomic_dec_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw sub i32 addrspace(3)* %ptr, i32 1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_dec_noret_i32_offset:
; SI: v_mov_b32_e32 [[NEGONE:v[0-9]+]], -1
; SI: ds_dec_u32 v{{[0-9]+}}, [[NEGONE]] offset:16
; SI: s_endpgm
define void @lds_atomic_dec_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw sub i32 addrspace(3)* %gep, i32 1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_and_noret_i32:
; SI: ds_and_b32
; SI: s_endpgm
define void @lds_atomic_and_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw and i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_and_noret_i32_offset:
; SI: ds_and_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_and_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw and i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_or_noret_i32:
; SI: ds_or_b32
; SI: s_endpgm
define void @lds_atomic_or_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw or i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_or_noret_i32_offset:
; SI: ds_or_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_or_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw or i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xor_noret_i32:
; SI: ds_xor_b32
; SI: s_endpgm
define void @lds_atomic_xor_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw xor i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xor_noret_i32_offset:
; SI: ds_xor_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
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
; SI: ds_min_i32
; SI: s_endpgm
define void @lds_atomic_min_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw min i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_min_noret_i32_offset:
; SI: ds_min_i32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_min_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw min i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_max_noret_i32:
; SI: ds_max_i32
; SI: s_endpgm
define void @lds_atomic_max_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw max i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_max_noret_i32_offset:
; SI: ds_max_i32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_max_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw max i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umin_noret_i32:
; SI: ds_min_u32
; SI: s_endpgm
define void @lds_atomic_umin_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw umin i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umin_noret_i32_offset:
; SI: ds_min_u32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_umin_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw umin i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umax_noret_i32:
; SI: ds_max_u32
; SI: s_endpgm
define void @lds_atomic_umax_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw umax i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umax_noret_i32_offset:
; SI: ds_max_u32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; SI: s_endpgm
define void @lds_atomic_umax_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw umax i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

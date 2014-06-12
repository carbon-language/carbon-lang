; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; FUNC-LABEL: @lds_atomic_xchg_ret_i64:
; SI: DS_WRXCHG_RTN_B64
; SI: S_ENDPGM
define void @lds_atomic_xchg_ret_i64(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw xchg i64 addrspace(3)* %ptr, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_xchg_ret_i64_offset:
; SI: DS_WRXCHG_RTN_B64 {{.*}} 0x20
; SI: S_ENDPGM
define void @lds_atomic_xchg_ret_i64_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i64 addrspace(3)* %ptr, i32 4
  %result = atomicrmw xchg i64 addrspace(3)* %gep, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_add_ret_i64:
; SI: DS_ADD_RTN_U64
; SI: S_ENDPGM
define void @lds_atomic_add_ret_i64(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw add i64 addrspace(3)* %ptr, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_add_ret_i64_offset:
; SI: S_LOAD_DWORD [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: S_MOV_B64 s{{\[}}[[LOSDATA:[0-9]+]]:[[HISDATA:[0-9]+]]{{\]}}, 9
; SI-DAG: V_MOV_B32_e32 v[[LOVDATA:[0-9]+]], s[[LOSDATA]]
; SI-DAG: V_MOV_B32_e32 v[[HIVDATA:[0-9]+]], s[[HISDATA]]
; SI-DAG: V_MOV_B32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; SI: DS_ADD_RTN_U64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[VPTR]], v{{\[}}[[LOVDATA]]:[[HIVDATA]]{{\]}}, 0x20, [M0]
; SI: BUFFER_STORE_DWORDX2 [[RESULT]],
; SI: S_ENDPGM
define void @lds_atomic_add_ret_i64_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i64 addrspace(3)* %ptr, i64 4
  %result = atomicrmw add i64 addrspace(3)* %gep, i64 9 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_inc_ret_i64:
; SI: S_MOV_B64 s{{\[}}[[LOSDATA:[0-9]+]]:[[HISDATA:[0-9]+]]{{\]}}, -1
; SI-DAG: V_MOV_B32_e32 v[[LOVDATA:[0-9]+]], s[[LOSDATA]]
; SI-DAG: V_MOV_B32_e32 v[[HIVDATA:[0-9]+]], s[[HISDATA]]
; SI: DS_INC_RTN_U64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[VPTR]], v{{\[}}[[LOVDATA]]:[[HIVDATA]]{{\]}},
; SI: BUFFER_STORE_DWORDX2 [[RESULT]],
; SI: S_ENDPGM
define void @lds_atomic_inc_ret_i64(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw add i64 addrspace(3)* %ptr, i64 1 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_inc_ret_i64_offset:
; SI: DS_INC_RTN_U64 {{.*}} 0x20
; SI: S_ENDPGM
define void @lds_atomic_inc_ret_i64_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i64 addrspace(3)* %ptr, i32 4
  %result = atomicrmw add i64 addrspace(3)* %gep, i64 1 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_sub_ret_i64:
; SI: DS_SUB_RTN_U64
; SI: S_ENDPGM
define void @lds_atomic_sub_ret_i64(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw sub i64 addrspace(3)* %ptr, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_sub_ret_i64_offset:
; SI: DS_SUB_RTN_U64 {{.*}} 0x20
; SI: S_ENDPGM
define void @lds_atomic_sub_ret_i64_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i64 addrspace(3)* %ptr, i32 4
  %result = atomicrmw sub i64 addrspace(3)* %gep, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_dec_ret_i64:
; SI: S_MOV_B64 s{{\[}}[[LOSDATA:[0-9]+]]:[[HISDATA:[0-9]+]]{{\]}}, -1
; SI-DAG: V_MOV_B32_e32 v[[LOVDATA:[0-9]+]], s[[LOSDATA]]
; SI-DAG: V_MOV_B32_e32 v[[HIVDATA:[0-9]+]], s[[HISDATA]]
; SI: DS_DEC_RTN_U64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[VPTR]], v{{\[}}[[LOVDATA]]:[[HIVDATA]]{{\]}},
; SI: BUFFER_STORE_DWORDX2 [[RESULT]],
; SI: S_ENDPGM
define void @lds_atomic_dec_ret_i64(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw sub i64 addrspace(3)* %ptr, i64 1 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_dec_ret_i64_offset:
; SI: DS_DEC_RTN_U64 {{.*}} 0x20
; SI: S_ENDPGM
define void @lds_atomic_dec_ret_i64_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i64 addrspace(3)* %ptr, i32 4
  %result = atomicrmw sub i64 addrspace(3)* %gep, i64 1 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_and_ret_i64:
; SI: DS_AND_RTN_B64
; SI: S_ENDPGM
define void @lds_atomic_and_ret_i64(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw and i64 addrspace(3)* %ptr, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_and_ret_i64_offset:
; SI: DS_AND_RTN_B64 {{.*}} 0x20
; SI: S_ENDPGM
define void @lds_atomic_and_ret_i64_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i64 addrspace(3)* %ptr, i32 4
  %result = atomicrmw and i64 addrspace(3)* %gep, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_or_ret_i64:
; SI: DS_OR_RTN_B64
; SI: S_ENDPGM
define void @lds_atomic_or_ret_i64(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw or i64 addrspace(3)* %ptr, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_or_ret_i64_offset:
; SI: DS_OR_RTN_B64 {{.*}} 0x20
; SI: S_ENDPGM
define void @lds_atomic_or_ret_i64_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i64 addrspace(3)* %ptr, i32 4
  %result = atomicrmw or i64 addrspace(3)* %gep, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_xor_ret_i64:
; SI: DS_XOR_RTN_B64
; SI: S_ENDPGM
define void @lds_atomic_xor_ret_i64(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw xor i64 addrspace(3)* %ptr, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_xor_ret_i64_offset:
; SI: DS_XOR_RTN_B64 {{.*}} 0x20
; SI: S_ENDPGM
define void @lds_atomic_xor_ret_i64_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i64 addrspace(3)* %ptr, i32 4
  %result = atomicrmw xor i64 addrspace(3)* %gep, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FIXME: There is no atomic nand instr
; XFUNC-LABEL: @lds_atomic_nand_ret_i64:uction, so we somehow need to expand this.
; define void @lds_atomic_nand_ret_i64(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
;   %result = atomicrmw nand i64 addrspace(3)* %ptr, i32 4 seq_cst
;   store i64 %result, i64 addrspace(1)* %out, align 8
;   ret void
; }

; FUNC-LABEL: @lds_atomic_min_ret_i64:
; SI: DS_MIN_RTN_I64
; SI: S_ENDPGM
define void @lds_atomic_min_ret_i64(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw min i64 addrspace(3)* %ptr, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_min_ret_i64_offset:
; SI: DS_MIN_RTN_I64 {{.*}} 0x20
; SI: S_ENDPGM
define void @lds_atomic_min_ret_i64_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i64 addrspace(3)* %ptr, i32 4
  %result = atomicrmw min i64 addrspace(3)* %gep, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_max_ret_i64:
; SI: DS_MAX_RTN_I64
; SI: S_ENDPGM
define void @lds_atomic_max_ret_i64(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw max i64 addrspace(3)* %ptr, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_max_ret_i64_offset:
; SI: DS_MAX_RTN_I64 {{.*}} 0x20
; SI: S_ENDPGM
define void @lds_atomic_max_ret_i64_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i64 addrspace(3)* %ptr, i32 4
  %result = atomicrmw max i64 addrspace(3)* %gep, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_umin_ret_i64:
; SI: DS_MIN_RTN_U64
; SI: S_ENDPGM
define void @lds_atomic_umin_ret_i64(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw umin i64 addrspace(3)* %ptr, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_umin_ret_i64_offset:
; SI: DS_MIN_RTN_U64 {{.*}} 0x20
; SI: S_ENDPGM
define void @lds_atomic_umin_ret_i64_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i64 addrspace(3)* %ptr, i32 4
  %result = atomicrmw umin i64 addrspace(3)* %gep, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_umax_ret_i64:
; SI: DS_MAX_RTN_U64
; SI: S_ENDPGM
define void @lds_atomic_umax_ret_i64(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw umax i64 addrspace(3)* %ptr, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @lds_atomic_umax_ret_i64_offset:
; SI: DS_MAX_RTN_U64 {{.*}} 0x20
; SI: S_ENDPGM
define void @lds_atomic_umax_ret_i64_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i64 addrspace(3)* %ptr, i32 4
  %result = atomicrmw umax i64 addrspace(3)* %gep, i64 4 seq_cst
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

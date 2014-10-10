; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=CI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}lds_atomic_cmpxchg_ret_i32_offset:
; SI: S_LOAD_DWORD [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: S_LOAD_DWORD [[SWAP:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI-DAG: V_MOV_B32_e32 [[VCMP:v[0-9]+]], 7
; SI-DAG: V_MOV_B32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; SI-DAG: V_MOV_B32_e32 [[VSWAP:v[0-9]+]], [[SWAP]]
; SI: DS_CMPST_RTN_B32 [[RESULT:v[0-9]+]], [[VPTR]], [[VCMP]], [[VSWAP]] offset:16 [M0]
; SI: S_ENDPGM
define void @lds_atomic_cmpxchg_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr, i32 %swap) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %pair = cmpxchg i32 addrspace(3)* %gep, i32 7, i32 %swap seq_cst monotonic
  %result = extractvalue { i32, i1 } %pair, 0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_cmpxchg_ret_i64_offset:
; SI: S_LOAD_DWORD [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: S_LOAD_DWORDX2 s{{\[}}[[LOSWAP:[0-9]+]]:[[HISWAP:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0xd
; SI: S_MOV_B64  s{{\[}}[[LOSCMP:[0-9]+]]:[[HISCMP:[0-9]+]]{{\]}}, 7
; SI-DAG: V_MOV_B32_e32 v[[LOVCMP:[0-9]+]], s[[LOSCMP]]
; SI-DAG: V_MOV_B32_e32 v[[HIVCMP:[0-9]+]], s[[HISCMP]]
; SI-DAG: V_MOV_B32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; SI-DAG: V_MOV_B32_e32 v[[LOSWAPV:[0-9]+]], s[[LOSWAP]]
; SI-DAG: V_MOV_B32_e32 v[[HISWAPV:[0-9]+]], s[[HISWAP]]
; SI: DS_CMPST_RTN_B64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[VPTR]], v{{\[}}[[LOVCMP]]:[[HIVCMP]]{{\]}}, v{{\[}}[[LOSWAPV]]:[[HISWAPV]]{{\]}} offset:32 [M0]
; SI: BUFFER_STORE_DWORDX2 [[RESULT]],
; SI: S_ENDPGM
define void @lds_atomic_cmpxchg_ret_i64_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr, i64 %swap) nounwind {
  %gep = getelementptr i64 addrspace(3)* %ptr, i32 4
  %pair = cmpxchg i64 addrspace(3)* %gep, i64 7, i64 %swap seq_cst monotonic
  %result = extractvalue { i64, i1 } %pair, 0
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_cmpxchg_ret_i32_bad_si_offset
; SI: DS_CMPST_RTN_B32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CI: DS_CMPST_RTN_B32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16 [M0]
; SI: S_ENDPGM
define void @lds_atomic_cmpxchg_ret_i32_bad_si_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr, i32 %swap, i32 %a, i32 %b) nounwind {
  %sub = sub i32 %a, %b
  %add = add i32 %sub, 4
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 %add
  %pair = cmpxchg i32 addrspace(3)* %gep, i32 7, i32 %swap seq_cst monotonic
  %result = extractvalue { i32, i1 } %pair, 0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_cmpxchg_noret_i32_offset:
; SI: S_LOAD_DWORD [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x9
; SI: S_LOAD_DWORD [[SWAP:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xa
; SI-DAG: V_MOV_B32_e32 [[VCMP:v[0-9]+]], 7
; SI-DAG: V_MOV_B32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; SI-DAG: V_MOV_B32_e32 [[VSWAP:v[0-9]+]], [[SWAP]]
; SI: DS_CMPST_B32 [[VPTR]], [[VCMP]], [[VSWAP]] offset:16 [M0]
; SI: S_ENDPGM
define void @lds_atomic_cmpxchg_noret_i32_offset(i32 addrspace(3)* %ptr, i32 %swap) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %pair = cmpxchg i32 addrspace(3)* %gep, i32 7, i32 %swap seq_cst monotonic
  %result = extractvalue { i32, i1 } %pair, 0
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_cmpxchg_noret_i64_offset:
; SI: S_LOAD_DWORD [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x9
; SI: S_LOAD_DWORDX2 s{{\[}}[[LOSWAP:[0-9]+]]:[[HISWAP:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: S_MOV_B64  s{{\[}}[[LOSCMP:[0-9]+]]:[[HISCMP:[0-9]+]]{{\]}}, 7
; SI-DAG: V_MOV_B32_e32 v[[LOVCMP:[0-9]+]], s[[LOSCMP]]
; SI-DAG: V_MOV_B32_e32 v[[HIVCMP:[0-9]+]], s[[HISCMP]]
; SI-DAG: V_MOV_B32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; SI-DAG: V_MOV_B32_e32 v[[LOSWAPV:[0-9]+]], s[[LOSWAP]]
; SI-DAG: V_MOV_B32_e32 v[[HISWAPV:[0-9]+]], s[[HISWAP]]
; SI: DS_CMPST_B64 [[VPTR]], v{{\[}}[[LOVCMP]]:[[HIVCMP]]{{\]}}, v{{\[}}[[LOSWAPV]]:[[HISWAPV]]{{\]}} offset:32 [M0]
; SI: S_ENDPGM
define void @lds_atomic_cmpxchg_noret_i64_offset(i64 addrspace(3)* %ptr, i64 %swap) nounwind {
  %gep = getelementptr i64 addrspace(3)* %ptr, i32 4
  %pair = cmpxchg i64 addrspace(3)* %gep, i64 7, i64 %swap seq_cst monotonic
  %result = extractvalue { i64, i1 } %pair, 0
  ret void
}

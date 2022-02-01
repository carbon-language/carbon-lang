; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; TODO: Add _RTN versions and merge with the GCN test

; FUNC-LABEL: {{^}}atomic_add_i32_offset:
; EG: MEM_RAT ATOMIC_ADD [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_add_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile add i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_i32_soffset:
; EG: MEM_RAT ATOMIC_ADD [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_add_i32_soffset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 9000
  %val = atomicrmw volatile add i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_i32_huge_offset:
; FIXME: looks like the offset is wrong
; EG: MEM_RAT ATOMIC_ADD [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_add_i32_huge_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 47224239175595

  %val = atomicrmw volatile add i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_i32_addr64_offset:
; EG: MEM_RAT ATOMIC_ADD [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_add_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile add i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_i32:
; EG: MEM_RAT ATOMIC_ADD [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_add_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile add i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_i32_addr64:
; EG: MEM_RAT ATOMIC_ADD [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_add_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile add i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_and_i32_offset:
; EG: MEM_RAT ATOMIC_AND [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_and_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile and i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_and_i32_addr64_offset:
; EG: MEM_RAT ATOMIC_AND [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_and_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile and i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_and_i32:
; EG: MEM_RAT ATOMIC_AND [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_and_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile and i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_and_i32_addr64:
; EG: MEM_RAT ATOMIC_AND [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_and_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile and i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_i32_offset:
; EG: MEM_RAT ATOMIC_SUB [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_sub_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile sub i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_i32_addr64_offset:
; EG: MEM_RAT ATOMIC_SUB [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_sub_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile sub i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_i32:
; EG: MEM_RAT ATOMIC_SUB [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_sub_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile sub i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_i32_addr64:
; EG: MEM_RAT ATOMIC_SUB [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_sub_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile sub i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_max_i32_offset:
; EG: MEM_RAT ATOMIC_MAX_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_max_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile max i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_max_i32_addr64_offset:
; EG: MEM_RAT ATOMIC_MAX_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_max_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile max i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_max_i32:
; EG: MEM_RAT ATOMIC_MAX_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_max_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile max i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_max_i32_addr64:
; EG: MEM_RAT ATOMIC_MAX_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_max_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile max i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_umax_i32_offset:
; EG: MEM_RAT ATOMIC_MAX_UINT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_umax_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile umax i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_umax_i32_addr64_offset:
; EG: MEM_RAT ATOMIC_MAX_UINT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_umax_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile umax i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_umax_i32:
; EG: MEM_RAT ATOMIC_MAX_UINT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_umax_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile umax i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_umax_i32_addr64:
; EG: MEM_RAT ATOMIC_MAX_UINT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_umax_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile umax i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_min_i32_offset:
; EG: MEM_RAT ATOMIC_MIN_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_min_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile min i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_min_i32_addr64_offset:
; EG: MEM_RAT ATOMIC_MIN_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_min_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile min i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_min_i32:
; EG: MEM_RAT ATOMIC_MIN_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_min_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile min i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_min_i32_addr64:
; EG: MEM_RAT ATOMIC_MIN_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_min_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile min i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_umin_i32_offset:
; EG: MEM_RAT ATOMIC_MIN_UINT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_umin_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile umin i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_umin_i32_addr64_offset:
; EG: MEM_RAT ATOMIC_MIN_UINT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_umin_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile umin i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_umin_i32:
; EG: MEM_RAT ATOMIC_MIN_UINT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_umin_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile umin i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_umin_i32_addr64:
; EG: MEM_RAT ATOMIC_MIN_UINT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_umin_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile umin i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_or_i32_offset:
; EG: MEM_RAT ATOMIC_OR [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_or_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile or i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_or_i32_addr64_offset:
; EG: MEM_RAT ATOMIC_OR [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_or_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile or i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_or_i32:
; EG: MEM_RAT ATOMIC_OR [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_or_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile or i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_or_i32_addr64:
; EG: MEM_RAT ATOMIC_OR [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_or_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile or i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_xchg_i32_offset:
; EG: MEM_RAT ATOMIC_XCHG_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_xchg_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile xchg i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_xchg_i32_addr64_offset:
; EG: MEM_RAT ATOMIC_XCHG_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_xchg_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile xchg i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_xchg_i32:
; EG: MEM_RAT ATOMIC_XCHG_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_xchg_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_xchg_i32_addr64:
; EG: MEM_RAT ATOMIC_XCHG_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_xchg_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile xchg i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_cmpxchg_i32_offset:
; EG: MEM_RAT ATOMIC_CMPXCHG_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_cmpxchg_i32_offset(i32 addrspace(1)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = cmpxchg volatile i32 addrspace(1)* %gep, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_cmpxchg_i32_addr64_offset:
; EG: MEM_RAT ATOMIC_CMPXCHG_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_cmpxchg_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index, i32 %old) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = cmpxchg volatile i32 addrspace(1)* %gep, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_cmpxchg_i32:
; EG: MEM_RAT ATOMIC_CMPXCHG_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_cmpxchg_i32(i32 addrspace(1)* %out, i32 %in, i32 %old) {
entry:
  %val = cmpxchg volatile i32 addrspace(1)* %out, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_cmpxchg_i32_addr64:
; EG: MEM_RAT ATOMIC_CMPXCHG_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_cmpxchg_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index, i32 %old) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = cmpxchg volatile i32 addrspace(1)* %ptr, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_xor_i32_offset:
; EG: MEM_RAT ATOMIC_XOR [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_xor_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile xor i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_xor_i32_addr64_offset:
; EG: MEM_RAT ATOMIC_XOR [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_xor_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile xor i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_xor_i32:
; EG: MEM_RAT ATOMIC_XOR [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_xor_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xor i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_xor_i32_addr64:
; EG: MEM_RAT ATOMIC_XOR [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Z
define amdgpu_kernel void @atomic_xor_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile xor i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_store_i32_offset:
; EG: MEM_RAT ATOMIC_XCHG_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Y
define amdgpu_kernel void @atomic_store_i32_offset(i32 %in, i32 addrspace(1)* %out) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  store atomic i32 %in, i32 addrspace(1)* %gep  seq_cst, align 4
  ret void
}

; FUNC-LABEL: {{^}}atomic_store_i32:
; EG: MEM_RAT ATOMIC_XCHG_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Y
define amdgpu_kernel void @atomic_store_i32(i32 %in, i32 addrspace(1)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(1)* %out seq_cst, align 4
  ret void
}

; FUNC-LABEL: {{^}}atomic_store_i32_addr64_offset:
; EG: MEM_RAT ATOMIC_XCHG_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Y
define amdgpu_kernel void @atomic_store_i32_addr64_offset(i32 %in, i32 addrspace(1)* %out, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  store atomic i32 %in, i32 addrspace(1)* %gep seq_cst, align 4
  ret void
}

; FUNC-LABEL: {{^}}atomic_store_i32_addr64:
; EG: MEM_RAT ATOMIC_XCHG_INT [[REG:T[0-9]+]]
; EG: MOV{{[ *]*}}[[REG]].X, KC0[2].Y
define amdgpu_kernel void @atomic_store_i32_addr64(i32 %in, i32 addrspace(1)* %out, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  store atomic i32 %in, i32 addrspace(1)* %ptr seq_cst, align 4
  ret void
}

; FUNC-LABEL: {{^}}atomic_inc_add
; EG: MEM_RAT ATOMIC_INC_UINT
define amdgpu_kernel void @atomic_inc_add(i32 addrspace(1)* %out) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile add i32 addrspace(1)* %gep, i32 1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_dec_add
; EG: MEM_RAT ATOMIC_DEC_UINT
define amdgpu_kernel void @atomic_dec_add(i32 addrspace(1)* %out) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile add i32 addrspace(1)* %gep, i32 -1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_inc_sub
; EG: MEM_RAT ATOMIC_INC_UINT
define amdgpu_kernel void @atomic_inc_sub(i32 addrspace(1)* %out) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile sub i32 addrspace(1)* %gep, i32 -1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_dec_sub
; EG: MEM_RAT ATOMIC_DEC_UINT
define amdgpu_kernel void @atomic_dec_sub(i32 addrspace(1)* %out) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile sub i32 addrspace(1)* %gep, i32 1 seq_cst
  ret void
}

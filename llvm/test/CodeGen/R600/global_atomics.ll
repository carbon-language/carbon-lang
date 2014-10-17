; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck --check-prefix=SI --check-prefix=FUNC %s

; FUNC-LABEL: {{^}}atomic_add_i32_offset:
; SI: BUFFER_ATOMIC_ADD v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10{{$}}
define void @atomic_add_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile add i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_i32_ret_offset:
; SI: BUFFER_ATOMIC_ADD [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10 glc {{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_add_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile add i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_i32_addr64_offset:
; SI: BUFFER_ATOMIC_ADD v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10{{$}}
define void @atomic_add_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile add i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_i32_ret_addr64_offset:
; SI: BUFFER_ATOMIC_ADD [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_add_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile add i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_i32:
; SI: BUFFER_ATOMIC_ADD v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_add_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0  = atomicrmw volatile add i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_i32_ret:
; SI: BUFFER_ATOMIC_ADD [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_add_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %0  = atomicrmw volatile add i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_i32_addr64:
; SI: BUFFER_ATOMIC_ADD v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
define void @atomic_add_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile add i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_i32_ret_addr64:
; SI: BUFFER_ATOMIC_ADD [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_add_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile add i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_and_i32_offset:
; SI: BUFFER_ATOMIC_AND v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10{{$}}
define void @atomic_and_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile and i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_and_i32_ret_offset:
; SI: BUFFER_ATOMIC_AND [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10 glc {{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_and_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile and i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_and_i32_addr64_offset:
; SI: BUFFER_ATOMIC_AND v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10{{$}}
define void @atomic_and_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile and i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_and_i32_ret_addr64_offset:
; SI: BUFFER_ATOMIC_AND [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_and_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile and i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_and_i32:
; SI: BUFFER_ATOMIC_AND v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_and_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0  = atomicrmw volatile and i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_and_i32_ret:
; SI: BUFFER_ATOMIC_AND [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_and_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %0  = atomicrmw volatile and i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_and_i32_addr64:
; SI: BUFFER_ATOMIC_AND v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
define void @atomic_and_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile and i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_and_i32_ret_addr64:
; SI: BUFFER_ATOMIC_AND [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_and_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile and i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_i32_offset:
; SI: BUFFER_ATOMIC_SUB v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10{{$}}
define void @atomic_sub_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile sub i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_i32_ret_offset:
; SI: BUFFER_ATOMIC_SUB [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10 glc {{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_sub_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile sub i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_i32_addr64_offset:
; SI: BUFFER_ATOMIC_SUB v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10{{$}}
define void @atomic_sub_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile sub i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_i32_ret_addr64_offset:
; SI: BUFFER_ATOMIC_SUB [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_sub_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile sub i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_i32:
; SI: BUFFER_ATOMIC_SUB v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_sub_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0  = atomicrmw volatile sub i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_i32_ret:
; SI: BUFFER_ATOMIC_SUB [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_sub_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %0  = atomicrmw volatile sub i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_i32_addr64:
; SI: BUFFER_ATOMIC_SUB v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
define void @atomic_sub_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile sub i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_i32_ret_addr64:
; SI: BUFFER_ATOMIC_SUB [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_sub_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile sub i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_max_i32_offset:
; SI: BUFFER_ATOMIC_SMAX v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10{{$}}
define void @atomic_max_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile max i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_max_i32_ret_offset:
; SI: BUFFER_ATOMIC_SMAX [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10 glc {{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_max_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile max i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_max_i32_addr64_offset:
; SI: BUFFER_ATOMIC_SMAX v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10{{$}}
define void @atomic_max_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile max i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_max_i32_ret_addr64_offset:
; SI: BUFFER_ATOMIC_SMAX [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_max_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile max i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_max_i32:
; SI: BUFFER_ATOMIC_SMAX v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_max_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0  = atomicrmw volatile max i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_max_i32_ret:
; SI: BUFFER_ATOMIC_SMAX [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_max_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %0  = atomicrmw volatile max i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_max_i32_addr64:
; SI: BUFFER_ATOMIC_SMAX v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
define void @atomic_max_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile max i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_max_i32_ret_addr64:
; SI: BUFFER_ATOMIC_SMAX [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_max_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile max i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_umax_i32_offset:
; SI: BUFFER_ATOMIC_UMAX v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10{{$}}
define void @atomic_umax_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile umax i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_umax_i32_ret_offset:
; SI: BUFFER_ATOMIC_UMAX [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10 glc {{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_umax_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile umax i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_umax_i32_addr64_offset:
; SI: BUFFER_ATOMIC_UMAX v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10{{$}}
define void @atomic_umax_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile umax i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_umax_i32_ret_addr64_offset:
; SI: BUFFER_ATOMIC_UMAX [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_umax_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile umax i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_umax_i32:
; SI: BUFFER_ATOMIC_UMAX v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_umax_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0  = atomicrmw volatile umax i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_umax_i32_ret:
; SI: BUFFER_ATOMIC_UMAX [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_umax_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %0  = atomicrmw volatile umax i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_umax_i32_addr64:
; SI: BUFFER_ATOMIC_UMAX v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
define void @atomic_umax_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile umax i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_umax_i32_ret_addr64:
; SI: BUFFER_ATOMIC_UMAX [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_umax_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile umax i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_min_i32_offset:
; SI: BUFFER_ATOMIC_SMIN v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10{{$}}
define void @atomic_min_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile min i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_min_i32_ret_offset:
; SI: BUFFER_ATOMIC_SMIN [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10 glc {{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_min_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile min i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_min_i32_addr64_offset:
; SI: BUFFER_ATOMIC_SMIN v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10{{$}}
define void @atomic_min_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile min i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_min_i32_ret_addr64_offset:
; SI: BUFFER_ATOMIC_SMIN [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_min_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile min i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_min_i32:
; SI: BUFFER_ATOMIC_SMIN v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_min_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0  = atomicrmw volatile min i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_min_i32_ret:
; SI: BUFFER_ATOMIC_SMIN [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_min_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %0  = atomicrmw volatile min i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_min_i32_addr64:
; SI: BUFFER_ATOMIC_SMIN v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
define void @atomic_min_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile min i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_min_i32_ret_addr64:
; SI: BUFFER_ATOMIC_SMIN [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_min_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile min i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_umin_i32_offset:
; SI: BUFFER_ATOMIC_UMIN v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10{{$}}
define void @atomic_umin_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile umin i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_umin_i32_ret_offset:
; SI: BUFFER_ATOMIC_UMIN [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10 glc {{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_umin_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile umin i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_umin_i32_addr64_offset:
; SI: BUFFER_ATOMIC_UMIN v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10{{$}}
define void @atomic_umin_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile umin i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_umin_i32_ret_addr64_offset:
; SI: BUFFER_ATOMIC_UMIN [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_umin_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile umin i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_umin_i32:
; SI: BUFFER_ATOMIC_UMIN v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_umin_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0  = atomicrmw volatile umin i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_umin_i32_ret:
; SI: BUFFER_ATOMIC_UMIN [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_umin_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %0  = atomicrmw volatile umin i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_umin_i32_addr64:
; SI: BUFFER_ATOMIC_UMIN v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
define void @atomic_umin_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile umin i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_umin_i32_ret_addr64:
; SI: BUFFER_ATOMIC_UMIN [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_umin_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile umin i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_or_i32_offset:
; SI: BUFFER_ATOMIC_OR v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10{{$}}
define void @atomic_or_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile or i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_or_i32_ret_offset:
; SI: BUFFER_ATOMIC_OR [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10 glc {{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_or_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile or i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_or_i32_addr64_offset:
; SI: BUFFER_ATOMIC_OR v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10{{$}}
define void @atomic_or_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile or i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_or_i32_ret_addr64_offset:
; SI: BUFFER_ATOMIC_OR [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_or_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile or i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_or_i32:
; SI: BUFFER_ATOMIC_OR v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_or_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0  = atomicrmw volatile or i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_or_i32_ret:
; SI: BUFFER_ATOMIC_OR [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_or_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %0  = atomicrmw volatile or i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_or_i32_addr64:
; SI: BUFFER_ATOMIC_OR v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
define void @atomic_or_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile or i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_or_i32_ret_addr64:
; SI: BUFFER_ATOMIC_OR [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_or_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile or i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_xchg_i32_offset:
; SI: BUFFER_ATOMIC_SWAP v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10{{$}}
define void @atomic_xchg_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile xchg i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_xchg_i32_ret_offset:
; SI: BUFFER_ATOMIC_SWAP [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10 glc {{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_xchg_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile xchg i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_xchg_i32_addr64_offset:
; SI: BUFFER_ATOMIC_SWAP v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10{{$}}
define void @atomic_xchg_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile xchg i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_xchg_i32_ret_addr64_offset:
; SI: BUFFER_ATOMIC_SWAP [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_xchg_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile xchg i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_xchg_i32:
; SI: BUFFER_ATOMIC_SWAP v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_xchg_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0  = atomicrmw volatile xchg i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_xchg_i32_ret:
; SI: BUFFER_ATOMIC_SWAP [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_xchg_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %0  = atomicrmw volatile xchg i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_xchg_i32_addr64:
; SI: BUFFER_ATOMIC_SWAP v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
define void @atomic_xchg_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile xchg i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_xchg_i32_ret_addr64:
; SI: BUFFER_ATOMIC_SWAP [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_xchg_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile xchg i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_xor_i32_offset:
; SI: BUFFER_ATOMIC_XOR v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10{{$}}
define void @atomic_xor_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile xor i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_xor_i32_ret_offset:
; SI: BUFFER_ATOMIC_XOR [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:0x10 glc {{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_xor_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32 addrspace(1)* %out, i32 4
  %0  = atomicrmw volatile xor i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_xor_i32_addr64_offset:
; SI: BUFFER_ATOMIC_XOR v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10{{$}}
define void @atomic_xor_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile xor i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_xor_i32_ret_addr64_offset:
; SI: BUFFER_ATOMIC_XOR [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:0x10 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_xor_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 4
  %0  = atomicrmw volatile xor i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_xor_i32:
; SI: BUFFER_ATOMIC_XOR v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_xor_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0  = atomicrmw volatile xor i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_xor_i32_ret:
; SI: BUFFER_ATOMIC_XOR [[RET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_xor_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %0  = atomicrmw volatile xor i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_xor_i32_addr64:
; SI: BUFFER_ATOMIC_XOR v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
define void @atomic_xor_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile xor i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_xor_i32_ret_addr64:
; SI: BUFFER_ATOMIC_XOR [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; SI: BUFFER_STORE_DWORD [[RET]]
define void @atomic_xor_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32 addrspace(1)* %out, i64 %index
  %0  = atomicrmw volatile xor i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %0, i32 addrspace(1)* %out2
  ret void
}

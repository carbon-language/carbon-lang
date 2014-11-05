; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s


; SI-LABEL: {{^}}global_copy_i1_to_i1:
; SI: buffer_load_ubyte
; SI: v_and_b32_e32 v{{[0-9]+}}, 1
; SI: buffer_store_byte
; SI: s_endpgm
define void @global_copy_i1_to_i1(i1 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1 addrspace(1)* %in
  store i1 %load, i1 addrspace(1)* %out, align 1
  ret void
}

; SI-LABEL: {{^}}global_sextload_i1_to_i32:
; XSI: BUFFER_LOAD_BYTE
; SI: buffer_store_dword
; SI: s_endpgm
define void @global_sextload_i1_to_i32(i32 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1 addrspace(1)* %in
  %ext = sext i1 %load to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}global_zextload_i1_to_i32:
; SI: buffer_load_ubyte
; SI: buffer_store_dword
; SI: s_endpgm
define void @global_zextload_i1_to_i32(i32 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1 addrspace(1)* %in
  %ext = zext i1 %load to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}global_sextload_i1_to_i64:
; XSI: BUFFER_LOAD_BYTE
; SI: buffer_store_dwordx2
; SI: s_endpgm
define void @global_sextload_i1_to_i64(i64 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1 addrspace(1)* %in
  %ext = sext i1 %load to i64
  store i64 %ext, i64 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}global_zextload_i1_to_i64:
; SI: buffer_load_ubyte
; SI: buffer_store_dwordx2
; SI: s_endpgm
define void @global_zextload_i1_to_i64(i64 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1 addrspace(1)* %in
  %ext = zext i1 %load to i64
  store i64 %ext, i64 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}i1_arg:
; SI: buffer_load_ubyte
; SI: v_and_b32_e32
; SI: buffer_store_byte
; SI: s_endpgm
define void @i1_arg(i1 addrspace(1)* %out, i1 %x) nounwind {
  store i1 %x, i1 addrspace(1)* %out, align 1
  ret void
}

; SI-LABEL: {{^}}i1_arg_zext_i32:
; SI: buffer_load_ubyte
; SI: buffer_store_dword
; SI: s_endpgm
define void @i1_arg_zext_i32(i32 addrspace(1)* %out, i1 %x) nounwind {
  %ext = zext i1 %x to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}i1_arg_zext_i64:
; SI: buffer_load_ubyte
; SI: buffer_store_dwordx2
; SI: s_endpgm
define void @i1_arg_zext_i64(i64 addrspace(1)* %out, i1 %x) nounwind {
  %ext = zext i1 %x to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}i1_arg_sext_i32:
; XSI: BUFFER_LOAD_BYTE
; SI: buffer_store_dword
; SI: s_endpgm
define void @i1_arg_sext_i32(i32 addrspace(1)* %out, i1 %x) nounwind {
  %ext = sext i1 %x to i32
  store i32 %ext, i32addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}i1_arg_sext_i64:
; XSI: BUFFER_LOAD_BYTE
; SI: buffer_store_dwordx2
; SI: s_endpgm
define void @i1_arg_sext_i64(i64 addrspace(1)* %out, i1 %x) nounwind {
  %ext = sext i1 %x to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

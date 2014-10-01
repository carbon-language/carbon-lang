; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s


; SI-LABEL: {{^}}global_copy_i1_to_i1:
; SI: BUFFER_LOAD_UBYTE
; SI: V_AND_B32_e32 v{{[0-9]+}}, 1
; SI: BUFFER_STORE_BYTE
; SI: S_ENDPGM
define void @global_copy_i1_to_i1(i1 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1 addrspace(1)* %in
  store i1 %load, i1 addrspace(1)* %out, align 1
  ret void
}

; SI-LABEL: {{^}}global_sextload_i1_to_i32:
; XSI: BUFFER_LOAD_BYTE
; SI: BUFFER_STORE_DWORD
; SI: S_ENDPGM
define void @global_sextload_i1_to_i32(i32 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1 addrspace(1)* %in
  %ext = sext i1 %load to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}global_zextload_i1_to_i32:
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_STORE_DWORD
; SI: S_ENDPGM
define void @global_zextload_i1_to_i32(i32 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1 addrspace(1)* %in
  %ext = zext i1 %load to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}global_sextload_i1_to_i64:
; XSI: BUFFER_LOAD_BYTE
; SI: BUFFER_STORE_DWORDX2
; SI: S_ENDPGM
define void @global_sextload_i1_to_i64(i64 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1 addrspace(1)* %in
  %ext = sext i1 %load to i64
  store i64 %ext, i64 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}global_zextload_i1_to_i64:
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_STORE_DWORDX2
; SI: S_ENDPGM
define void @global_zextload_i1_to_i64(i64 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1 addrspace(1)* %in
  %ext = zext i1 %load to i64
  store i64 %ext, i64 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}i1_arg:
; SI: BUFFER_LOAD_UBYTE
; SI: V_AND_B32_e32
; SI: BUFFER_STORE_BYTE
; SI: S_ENDPGM
define void @i1_arg(i1 addrspace(1)* %out, i1 %x) nounwind {
  store i1 %x, i1 addrspace(1)* %out, align 1
  ret void
}

; SI-LABEL: {{^}}i1_arg_zext_i32:
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_STORE_DWORD
; SI: S_ENDPGM
define void @i1_arg_zext_i32(i32 addrspace(1)* %out, i1 %x) nounwind {
  %ext = zext i1 %x to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}i1_arg_zext_i64:
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_STORE_DWORDX2
; SI: S_ENDPGM
define void @i1_arg_zext_i64(i64 addrspace(1)* %out, i1 %x) nounwind {
  %ext = zext i1 %x to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}i1_arg_sext_i32:
; XSI: BUFFER_LOAD_BYTE
; SI: BUFFER_STORE_DWORD
; SI: S_ENDPGM
define void @i1_arg_sext_i32(i32 addrspace(1)* %out, i1 %x) nounwind {
  %ext = sext i1 %x to i32
  store i32 %ext, i32addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}i1_arg_sext_i64:
; XSI: BUFFER_LOAD_BYTE
; SI: BUFFER_STORE_DWORDX2
; SI: S_ENDPGM
define void @i1_arg_sext_i64(i64 addrspace(1)* %out, i1 %x) nounwind {
  %ext = sext i1 %x to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

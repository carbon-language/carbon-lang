; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s

; FIXME: This is probably wrong. This probably needs to expand to 8-bit reads and writes.
; SI-LABEL: @unaligned_load_store_i32:
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_WRITE_B32
; SI: S_ENDPGM
define void @unaligned_load_store_i32(i32 addrspace(3)* %p, i32 addrspace(3)* %r) nounwind {
  %v = load i32 addrspace(3)* %p, align 1
  store i32 %v, i32 addrspace(3)* %r, align 1
  ret void
}

; SI-LABEL: @unaligned_load_store_v4i32:
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_WRITE_B32
; SI: DS_WRITE_B32
; SI: DS_WRITE_B32
; SI: DS_WRITE_B32
; SI: S_ENDPGM
define void @unaligned_load_store_v4i32(<4 x i32> addrspace(3)* %p, <4 x i32> addrspace(3)* %r) nounwind {
  %v = load <4 x i32> addrspace(3)* %p, align 1
  store <4 x i32> %v, <4 x i32> addrspace(3)* %r, align 1
  ret void
}

; FIXME: This should use ds_read2_b32
; SI-LABEL: @load_lds_i64_align_4
; SI: DS_READ_B64
; SI: S_ENDPGM
define void @load_lds_i64_align_4(i64 addrspace(1)* nocapture %out, i64 addrspace(3)* %in) #0 {
  %val = load i64 addrspace(3)* %in, align 4
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; FIXME: Need to fix this case.
; define void @load_lds_i64_align_1(i64 addrspace(1)* nocapture %out, i64 addrspace(3)* %in) #0 {
;   %val = load i64 addrspace(3)* %in, align 1
;   store i64 %val, i64 addrspace(1)* %out, align 8
;   ret void
; }

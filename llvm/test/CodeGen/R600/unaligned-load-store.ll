; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s

; FIXME: This is probably wrong. This probably needs to expand to 8-bit reads and writes.
; SI-LABEL: {{^}}unaligned_load_store_i32:
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_WRITE_B32
; SI: S_ENDPGM
define void @unaligned_load_store_i32(i32 addrspace(3)* %p, i32 addrspace(3)* %r) nounwind {
  %v = load i32 addrspace(3)* %p, align 1
  store i32 %v, i32 addrspace(3)* %r, align 1
  ret void
}

; SI-LABEL: {{^}}unaligned_load_store_v4i32:
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

; SI-LABEL: {{^}}load_lds_i64_align_4:
; SI: DS_READ2_B32
; SI: S_ENDPGM
define void @load_lds_i64_align_4(i64 addrspace(1)* nocapture %out, i64 addrspace(3)* %in) #0 {
  %val = load i64 addrspace(3)* %in, align 4
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}load_lds_i64_align_4_with_offset
; SI: DS_READ2_B32 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]}} offset0:8 offset1:9
; SI: S_ENDPGM
define void @load_lds_i64_align_4_with_offset(i64 addrspace(1)* nocapture %out, i64 addrspace(3)* %in) #0 {
  %ptr = getelementptr i64 addrspace(3)* %in, i32 4
  %val = load i64 addrspace(3)* %ptr, align 4
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}load_lds_i64_align_4_with_split_offset:
; The tests for the case where the lo offset is 8-bits, but the hi offset is 9-bits
; SI: DS_READ2_B32 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]}} offset0:0 offset1:1
; SI: S_ENDPGM
define void @load_lds_i64_align_4_with_split_offset(i64 addrspace(1)* nocapture %out, i64 addrspace(3)* %in) #0 {
  %ptr = bitcast i64 addrspace(3)* %in to i32 addrspace(3)*
  %ptr255 = getelementptr i32 addrspace(3)* %ptr, i32 255
  %ptri64 = bitcast i32 addrspace(3)* %ptr255 to i64 addrspace(3)*
  %val = load i64 addrspace(3)* %ptri64, align 4
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; FIXME: Need to fix this case.
; define void @load_lds_i64_align_1(i64 addrspace(1)* nocapture %out, i64 addrspace(3)* %in) #0 {
;   %val = load i64 addrspace(3)* %in, align 1
;   store i64 %val, i64 addrspace(1)* %out, align 8
;   ret void
; }

; SI-LABEL: {{^}}store_lds_i64_align_4:
; SI: DS_WRITE2_B32
; SI: S_ENDPGM
define void @store_lds_i64_align_4(i64 addrspace(3)* %out, i64 %val) #0 {
  store i64 %val, i64 addrspace(3)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}store_lds_i64_align_4_with_offset
; SI: DS_WRITE2_B32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:8 offset1:9
; SI: S_ENDPGM
define void @store_lds_i64_align_4_with_offset(i64 addrspace(3)* %out) #0 {
  %ptr = getelementptr i64 addrspace(3)* %out, i32 4
  store i64 0, i64 addrspace(3)* %ptr, align 4
  ret void
}

; SI-LABEL: {{^}}store_lds_i64_align_4_with_split_offset:
; The tests for the case where the lo offset is 8-bits, but the hi offset is 9-bits
; SI: DS_WRITE2_B32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:0 offset1:1
; SI: S_ENDPGM
define void @store_lds_i64_align_4_with_split_offset(i64 addrspace(3)* %out) #0 {
  %ptr = bitcast i64 addrspace(3)* %out to i32 addrspace(3)*
  %ptr255 = getelementptr i32 addrspace(3)* %ptr, i32 255
  %ptri64 = bitcast i32 addrspace(3)* %ptr255 to i64 addrspace(3)*
  store i64 0, i64 addrspace(3)* %out, align 4
  ret void
}

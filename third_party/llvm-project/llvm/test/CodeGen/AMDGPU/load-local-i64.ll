; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -mattr=-enable-ds128 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-enable-ds128 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-enable-ds128 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9,FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefixes=EG,FUNC %s

; Testing for ds_read/write_b128
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs -mattr=+enable-ds128 < %s | FileCheck -check-prefixes=CIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -mattr=+enable-ds128 < %s | FileCheck -check-prefixes=CIVI,FUNC %s

; FUNC-LABEL: {{^}}local_load_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read_b64 [[VAL:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}{{$}}
; GCN: ds_write_b64 v{{[0-9]+}}, [[VAL]]

; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_i64(i64 addrspace(3)* %out, i64 addrspace(3)* %in) #0 {
  %ld = load i64, i64 addrspace(3)* %in
  store i64 %ld, i64 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v2i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v2i64(<2 x i64> addrspace(3)* %out, <2 x i64> addrspace(3)* %in) #0 {
entry:
  %ld = load <2 x i64>, <2 x i64> addrspace(3)* %in
  store <2 x i64> %ld, <2 x i64> addrspace(3)* %out
  ret void
}

; Tests if ds_read/write_b128 gets generated for the 16 byte aligned load.
; FUNC-LABEL: {{^}}local_load_v2i64_to_128:

; CIVI: ds_read_b128
; CIVI: ds_write_b128

define amdgpu_kernel void @local_load_v2i64_to_128(<2 x i64> addrspace(3)* %out, <2 x i64> addrspace(3)* %in) {
entry:
  %ld = load <2 x i64>, <2 x i64> addrspace(3)* %in, align 16
  store <2 x i64> %ld, <2 x i64> addrspace(3)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}local_load_v3i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: ds_read2_b64
; GCN-DAG: ds_read_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v3i64(<3 x i64> addrspace(3)* %out, <3 x i64> addrspace(3)* %in) #0 {
entry:
  %ld = load <3 x i64>, <3 x i64> addrspace(3)* %in
  store <3 x i64> %ld, <3 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v4i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b64
; GCN: ds_read2_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v4i64(<4 x i64> addrspace(3)* %out, <4 x i64> addrspace(3)* %in) #0 {
entry:
  %ld = load <4 x i64>, <4 x i64> addrspace(3)* %in
  store <4 x i64> %ld, <4 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v8i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b64
; GCN: ds_read2_b64
; GCN: ds_read2_b64
; GCN: ds_read2_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v8i64(<8 x i64> addrspace(3)* %out, <8 x i64> addrspace(3)* %in) #0 {
entry:
  %ld = load <8 x i64>, <8 x i64> addrspace(3)* %in
  store <8 x i64> %ld, <8 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v16i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b64
; GCN: ds_read2_b64
; GCN: ds_read2_b64
; GCN: ds_read2_b64
; GCN: ds_read2_b64
; GCN: ds_read2_b64
; GCN: ds_read2_b64
; GCN: ds_read2_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v16i64(<16 x i64> addrspace(3)* %out, <16 x i64> addrspace(3)* %in) #0 {
entry:
  %ld = load <16 x i64>, <16 x i64> addrspace(3)* %in
  store <16 x i64> %ld, <16 x i64> addrspace(3)* %out
  ret void
}

attributes #0 = { nounwind }

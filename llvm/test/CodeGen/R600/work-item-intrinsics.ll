; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s


; FUNC-LABEL: {{^}}ngroups_x:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[0].X

; SI: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0
; SI: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI: BUFFER_STORE_DWORD [[VVAL]]
define void @ngroups_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ngroups_y:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[0].Y

; SI: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0x1
; SI: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI: BUFFER_STORE_DWORD [[VVAL]]
define void @ngroups_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ngroups_z:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[0].Z

; SI: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0x2
; SI: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI: BUFFER_STORE_DWORD [[VVAL]]
define void @ngroups_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_size_x:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[0].W

; SI: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0x3
; SI: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI: BUFFER_STORE_DWORD [[VVAL]]
define void @global_size_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_size_y:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[1].X

; SI: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0x4
; SI: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI: BUFFER_STORE_DWORD [[VVAL]]
define void @global_size_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_size_z:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[1].Y

; SI: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0x5
; SI: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI: BUFFER_STORE_DWORD [[VVAL]]
define void @global_size_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_x:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[1].Z

; SI: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0x6
; SI: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI: BUFFER_STORE_DWORD [[VVAL]]
define void @local_size_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_y:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[1].W

; SI: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0x7
; SI: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI: BUFFER_STORE_DWORD [[VVAL]]
define void @local_size_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_z:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[2].X

; SI: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0x8
; SI: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI: BUFFER_STORE_DWORD [[VVAL]]
define void @local_size_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}get_work_dim:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[2].Z

; SI: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0xb
; SI: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI: BUFFER_STORE_DWORD [[VVAL]]
define void @get_work_dim (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.AMDGPU.read.workdim() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; The tgid values are stored in sgprs offset by the number of user sgprs.
; Currently we always use exactly 2 user sgprs for the pointer to the
; kernel arguments, but this may change in the future.

; FUNC-LABEL: {{^}}tgid_x:
; SI: V_MOV_B32_e32 [[VVAL:v[0-9]+]], s4
; SI: BUFFER_STORE_DWORD [[VVAL]]
define void @tgid_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tgid_y:
; SI: V_MOV_B32_e32 [[VVAL:v[0-9]+]], s5
; SI: BUFFER_STORE_DWORD [[VVAL]]
define void @tgid_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tgid_z:
; SI: V_MOV_B32_e32 [[VVAL:v[0-9]+]], s6
; SI: BUFFER_STORE_DWORD [[VVAL]]
define void @tgid_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tidig_x:
; SI: BUFFER_STORE_DWORD v0
define void @tidig_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tidig_y:
; SI: BUFFER_STORE_DWORD v1
define void @tidig_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tidig_z:
; SI: BUFFER_STORE_DWORD v2
define void @tidig_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

declare i32 @llvm.r600.read.ngroups.x() #0
declare i32 @llvm.r600.read.ngroups.y() #0
declare i32 @llvm.r600.read.ngroups.z() #0

declare i32 @llvm.r600.read.global.size.x() #0
declare i32 @llvm.r600.read.global.size.y() #0
declare i32 @llvm.r600.read.global.size.z() #0

declare i32 @llvm.r600.read.local.size.x() #0
declare i32 @llvm.r600.read.local.size.y() #0
declare i32 @llvm.r600.read.local.size.z() #0

declare i32 @llvm.r600.read.tgid.x() #0
declare i32 @llvm.r600.read.tgid.y() #0
declare i32 @llvm.r600.read.tgid.z() #0

declare i32 @llvm.r600.read.tidig.x() #0
declare i32 @llvm.r600.read.tidig.y() #0
declare i32 @llvm.r600.read.tidig.z() #0

declare i32 @llvm.AMDGPU.read.workdim() #0

attributes #0 = { readnone }

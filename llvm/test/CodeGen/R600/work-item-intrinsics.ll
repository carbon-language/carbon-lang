; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=R600-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck --check-prefix=SI-CHECK %s

; R600-CHECK: @ngroups_x
; R600-CHECK: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV [[VAL]], KC0[0].X
; SI-CHECK: @ngroups_x
; SI-CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0
; SI-CHECK: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @ngroups_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; R600-CHECK: @ngroups_y
; R600-CHECK: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV [[VAL]], KC0[0].Y
; SI-CHECK: @ngroups_y
; SI-CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0x1
; SI-CHECK: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @ngroups_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; R600-CHECK: @ngroups_z
; R600-CHECK: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV [[VAL]], KC0[0].Z
; SI-CHECK: @ngroups_z
; SI-CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0x2
; SI-CHECK: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @ngroups_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; R600-CHECK: @global_size_x
; R600-CHECK: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV [[VAL]], KC0[0].W
; SI-CHECK: @global_size_x
; SI-CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0x3
; SI-CHECK: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @global_size_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; R600-CHECK: @global_size_y
; R600-CHECK: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV [[VAL]], KC0[1].X
; SI-CHECK: @global_size_y
; SI-CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0x4
; SI-CHECK: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @global_size_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; R600-CHECK: @global_size_z
; R600-CHECK: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV [[VAL]], KC0[1].Y
; SI-CHECK: @global_size_z
; SI-CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0x5
; SI-CHECK: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @global_size_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; R600-CHECK: @local_size_x
; R600-CHECK: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV [[VAL]], KC0[1].Z
; SI-CHECK: @local_size_x
; SI-CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0x6
; SI-CHECK: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @local_size_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; R600-CHECK: @local_size_y
; R600-CHECK: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV [[VAL]], KC0[1].W
; SI-CHECK: @local_size_y
; SI-CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0x7
; SI-CHECK: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @local_size_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; R600-CHECK: @local_size_z
; R600-CHECK: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV [[VAL]], KC0[2].X
; SI-CHECK: @local_size_z
; SI-CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]], s[0:1], 0x8
; SI-CHECK: V_MOV_B32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @local_size_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; The tgid values are stored in ss offset by the number of user ss.
; Currently we always use exactly 2 user ss for the pointer to the
; kernel arguments, but this may change in the future.

; SI-CHECK: @tgid_x
; SI-CHECK: V_MOV_B32_e32 [[VVAL:v[0-9]+]], s2
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @tgid_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; SI-CHECK: @tgid_y
; SI-CHECK: V_MOV_B32_e32 [[VVAL:v[0-9]+]], s3
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @tgid_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; SI-CHECK: @tgid_z
; SI-CHECK: V_MOV_B32_e32 [[VVAL:v[0-9]+]], s4
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @tgid_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; SI-CHECK: @tidig_x
; SI-CHECK: BUFFER_STORE_DWORD v0
define void @tidig_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; SI-CHECK: @tidig_y
; SI-CHECK: BUFFER_STORE_DWORD v1
define void @tidig_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; SI-CHECK: @tidig_z
; SI-CHECK: BUFFER_STORE_DWORD v2
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

attributes #0 = { readnone }

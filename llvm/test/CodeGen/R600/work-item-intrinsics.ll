; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=R600-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=SI | FileCheck --check-prefix=SI-CHECK %s

; R600-CHECK: @ngroups_x
; R600-CHECK: RAT_WRITE_CACHELESS_32_eg [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV * [[VAL]], KC0[0].X
; SI-CHECK: @ngroups_x
; SI-CHECK: S_LOAD_DWORD [[VAL:SGPR[0-9]+]], SGPR0_SGPR1, 0
; SI-CHECK: V_MOV_B32_e32 [[VVAL:VGPR[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @ngroups_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; R600-CHECK: @ngroups_y
; R600-CHECK: RAT_WRITE_CACHELESS_32_eg [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV * [[VAL]], KC0[0].Y
; SI-CHECK: @ngroups_y
; SI-CHECK: S_LOAD_DWORD [[VAL:SGPR[0-9]+]], SGPR0_SGPR1, 1
; SI-CHECK: V_MOV_B32_e32 [[VVAL:VGPR[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @ngroups_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; R600-CHECK: @ngroups_z
; R600-CHECK: RAT_WRITE_CACHELESS_32_eg [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV * [[VAL]], KC0[0].Z
; SI-CHECK: @ngroups_z
; SI-CHECK: S_LOAD_DWORD [[VAL:SGPR[0-9]+]], SGPR0_SGPR1, 2
; SI-CHECK: V_MOV_B32_e32 [[VVAL:VGPR[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @ngroups_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; R600-CHECK: @global_size_x
; R600-CHECK: RAT_WRITE_CACHELESS_32_eg [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV * [[VAL]], KC0[0].W
; SI-CHECK: @global_size_x
; SI-CHECK: S_LOAD_DWORD [[VAL:SGPR[0-9]+]], SGPR0_SGPR1, 3
; SI-CHECK: V_MOV_B32_e32 [[VVAL:VGPR[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @global_size_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; R600-CHECK: @global_size_y
; R600-CHECK: RAT_WRITE_CACHELESS_32_eg [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV * [[VAL]], KC0[1].X
; SI-CHECK: @global_size_y
; SI-CHECK: S_LOAD_DWORD [[VAL:SGPR[0-9]+]], SGPR0_SGPR1, 4
; SI-CHECK: V_MOV_B32_e32 [[VVAL:VGPR[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @global_size_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; R600-CHECK: @global_size_z
; R600-CHECK: RAT_WRITE_CACHELESS_32_eg [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV * [[VAL]], KC0[1].Y
; SI-CHECK: @global_size_z
; SI-CHECK: S_LOAD_DWORD [[VAL:SGPR[0-9]+]], SGPR0_SGPR1, 5
; SI-CHECK: V_MOV_B32_e32 [[VVAL:VGPR[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @global_size_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; R600-CHECK: @local_size_x
; R600-CHECK: RAT_WRITE_CACHELESS_32_eg [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV * [[VAL]], KC0[1].Z
; SI-CHECK: @local_size_x
; SI-CHECK: S_LOAD_DWORD [[VAL:SGPR[0-9]+]], SGPR0_SGPR1, 6
; SI-CHECK: V_MOV_B32_e32 [[VVAL:VGPR[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @local_size_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; R600-CHECK: @local_size_y
; R600-CHECK: RAT_WRITE_CACHELESS_32_eg [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV * [[VAL]], KC0[1].W
; SI-CHECK: @local_size_y
; SI-CHECK: S_LOAD_DWORD [[VAL:SGPR[0-9]+]], SGPR0_SGPR1, 7
; SI-CHECK: V_MOV_B32_e32 [[VVAL:VGPR[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @local_size_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; R600-CHECK: @local_size_z
; R600-CHECK: RAT_WRITE_CACHELESS_32_eg [[VAL:T[0-9]+\.X]]
; R600-CHECK: MOV * [[VAL]], KC0[2].X
; SI-CHECK: @local_size_z
; SI-CHECK: S_LOAD_DWORD [[VAL:SGPR[0-9]+]], SGPR0_SGPR1, 8
; SI-CHECK: V_MOV_B32_e32 [[VVAL:VGPR[0-9]+]], [[VAL]]
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @local_size_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; The tgid values are stored in SGPRs offset by the number of user SGPRs.
; Currently we always use exactly 2 user SGPRs for the pointer to the
; kernel arguments, but this may change in the future.

; SI-CHECK: @tgid_x
; SI-CHECK: V_MOV_B32_e32 [[VVAL:VGPR[0-9]+]], SGPR2
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @tgid_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; SI-CHECK: @tgid_y
; SI-CHECK: V_MOV_B32_e32 [[VVAL:VGPR[0-9]+]], SGPR3
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @tgid_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; SI-CHECK: @tgid_z
; SI-CHECK: V_MOV_B32_e32 [[VVAL:VGPR[0-9]+]], SGPR4
; SI-CHECK: BUFFER_STORE_DWORD [[VVAL]]
define void @tgid_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; SI-CHECK: @tidig_x
; SI-CHECK: BUFFER_STORE_DWORD VGPR0
define void @tidig_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; SI-CHECK: @tidig_y
; SI-CHECK: BUFFER_STORE_DWORD VGPR1
define void @tidig_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; SI-CHECK: @tidig_z
; SI-CHECK: BUFFER_STORE_DWORD VGPR2
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

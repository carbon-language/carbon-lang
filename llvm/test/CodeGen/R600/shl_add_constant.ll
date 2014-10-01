; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare i32 @llvm.r600.read.tidig.x() #1

; Test with inline immediate

; FUNC-LABEL: {{^}}shl_2_add_9_i32:
; SI: V_LSHLREV_B32_e32  [[REG:v[0-9]+]], 2, {{v[0-9]+}}
; SI: V_ADD_I32_e32 [[RESULT:v[0-9]+]], 36, [[REG]]
; SI: BUFFER_STORE_DWORD [[RESULT]]
; SI: S_ENDPGM
define void @shl_2_add_9_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid.x = tail call i32 @llvm.r600.read.tidig.x() #1
  %ptr = getelementptr i32 addrspace(1)* %in, i32 %tid.x
  %val = load i32 addrspace(1)* %ptr, align 4
  %add = add i32 %val, 9
  %result = shl i32 %add, 2
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}shl_2_add_9_i32_2_add_uses:
; SI-DAG: V_ADD_I32_e32 [[ADDREG:v[0-9]+]], 9, {{v[0-9]+}}
; SI-DAG: V_LSHLREV_B32_e32 [[SHLREG:v[0-9]+]], 2, {{v[0-9]+}}
; SI-DAG: BUFFER_STORE_DWORD [[ADDREG]]
; SI-DAG: BUFFER_STORE_DWORD [[SHLREG]]
; SI: S_ENDPGM
define void @shl_2_add_9_i32_2_add_uses(i32 addrspace(1)* %out0, i32 addrspace(1)* %out1, i32 addrspace(1)* %in) #0 {
  %tid.x = tail call i32 @llvm.r600.read.tidig.x() #1
  %ptr = getelementptr i32 addrspace(1)* %in, i32 %tid.x
  %val = load i32 addrspace(1)* %ptr, align 4
  %add = add i32 %val, 9
  %result = shl i32 %add, 2
  store i32 %result, i32 addrspace(1)* %out0, align 4
  store i32 %add, i32 addrspace(1)* %out1, align 4
  ret void
}

; Test with add literal constant

; FUNC-LABEL: {{^}}shl_2_add_999_i32:
; SI: V_LSHLREV_B32_e32  [[REG:v[0-9]+]], 2, {{v[0-9]+}}
; SI: V_ADD_I32_e32 [[RESULT:v[0-9]+]], 0xf9c, [[REG]]
; SI: BUFFER_STORE_DWORD [[RESULT]]
; SI: S_ENDPGM
define void @shl_2_add_999_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid.x = tail call i32 @llvm.r600.read.tidig.x() #1
  %ptr = getelementptr i32 addrspace(1)* %in, i32 %tid.x
  %val = load i32 addrspace(1)* %ptr, align 4
  %shl = add i32 %val, 999
  %result = shl i32 %shl, 2
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_add_shl_add_constant:
; SI-DAG: S_LOAD_DWORD [[X:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: S_LOAD_DWORD [[Y:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI: S_LSHL_B32 [[SHL3:s[0-9]+]], [[X]], 3
; SI: S_ADD_I32 [[TMP:s[0-9]+]], [[SHL3]], [[Y]]
; SI: S_ADD_I32 [[RESULT:s[0-9]+]], [[TMP]], 0x3d8
; SI: V_MOV_B32_e32 [[VRESULT:v[0-9]+]], [[RESULT]]
; SI: BUFFER_STORE_DWORD [[VRESULT]]
define void @test_add_shl_add_constant(i32 addrspace(1)* %out, i32 %x, i32 %y) #0 {
  %add.0 = add i32 %x, 123
  %shl = shl i32 %add.0, 3
  %add.1 = add i32 %shl, %y
   store i32 %add.1, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_add_shl_add_constant_inv:
; SI-DAG: S_LOAD_DWORD [[X:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: S_LOAD_DWORD [[Y:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI: S_LSHL_B32 [[SHL3:s[0-9]+]], [[X]], 3
; SI: S_ADD_I32 [[TMP:s[0-9]+]], [[SHL3]], [[Y]]
; SI: S_ADD_I32 [[RESULT:s[0-9]+]], [[TMP]], 0x3d8
; SI: V_MOV_B32_e32 [[VRESULT:v[0-9]+]], [[RESULT]]
; SI: BUFFER_STORE_DWORD [[VRESULT]]

define void @test_add_shl_add_constant_inv(i32 addrspace(1)* %out, i32 %x, i32 %y) #0 {
  %add.0 = add i32 %x, 123
  %shl = shl i32 %add.0, 3
  %add.1 = add i32 %y, %shl
  store i32 %add.1, i32 addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

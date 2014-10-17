; RUN: llc -march=r600 -mcpu=verde -verify-machineinstrs < %s | FileCheck %s

; Use a 64-bit value with lo bits that can be represented as an inline constant
; CHECK-LABEL: {{^}}i64_imm_inline_lo:
; CHECK: S_MOV_B32 [[LO:s[0-9]+]], 5
; CHECK: V_MOV_B32_e32 v[[LO_VGPR:[0-9]+]], [[LO]]
; CHECK: BUFFER_STORE_DWORDX2 v{{\[}}[[LO_VGPR]]:
define void @i64_imm_inline_lo(i64 addrspace(1) *%out) {
entry:
  store i64 1311768464867721221, i64 addrspace(1) *%out ; 0x1234567800000005
  ret void
}

; Use a 64-bit value with hi bits that can be represented as an inline constant
; CHECK-LABEL: {{^}}i64_imm_inline_hi:
; CHECK: S_MOV_B32 [[HI:s[0-9]+]], 5
; CHECK: V_MOV_B32_e32 v[[HI_VGPR:[0-9]+]], [[HI]]
; CHECK: BUFFER_STORE_DWORDX2 v{{\[[0-9]+:}}[[HI_VGPR]]
define void @i64_imm_inline_hi(i64 addrspace(1) *%out) {
entry:
  store i64 21780256376, i64 addrspace(1) *%out ; 0x0000000512345678
  ret void
}

; CHECK-LABEL: {{^}}store_inline_imm_0.0_f32
; CHECK: V_MOV_B32_e32 [[REG:v[0-9]+]], 0{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @store_inline_imm_0.0_f32(float addrspace(1)* %out) {
  store float 0.0, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}store_inline_imm_0.5_f32
; CHECK: V_MOV_B32_e32 [[REG:v[0-9]+]], 0.5{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @store_inline_imm_0.5_f32(float addrspace(1)* %out) {
  store float 0.5, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}store_inline_imm_m_0.5_f32
; CHECK: V_MOV_B32_e32 [[REG:v[0-9]+]], -0.5{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @store_inline_imm_m_0.5_f32(float addrspace(1)* %out) {
  store float -0.5, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}store_inline_imm_1.0_f32
; CHECK: V_MOV_B32_e32 [[REG:v[0-9]+]], 1.0{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @store_inline_imm_1.0_f32(float addrspace(1)* %out) {
  store float 1.0, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}store_inline_imm_m_1.0_f32
; CHECK: V_MOV_B32_e32 [[REG:v[0-9]+]], -1.0{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @store_inline_imm_m_1.0_f32(float addrspace(1)* %out) {
  store float -1.0, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}store_inline_imm_2.0_f32
; CHECK: V_MOV_B32_e32 [[REG:v[0-9]+]], 2.0{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @store_inline_imm_2.0_f32(float addrspace(1)* %out) {
  store float 2.0, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}store_inline_imm_m_2.0_f32
; CHECK: V_MOV_B32_e32 [[REG:v[0-9]+]], -2.0{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @store_inline_imm_m_2.0_f32(float addrspace(1)* %out) {
  store float -2.0, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}store_inline_imm_4.0_f32
; CHECK: V_MOV_B32_e32 [[REG:v[0-9]+]], 4.0{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @store_inline_imm_4.0_f32(float addrspace(1)* %out) {
  store float 4.0, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}store_inline_imm_m_4.0_f32
; CHECK: V_MOV_B32_e32 [[REG:v[0-9]+]], -4.0{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @store_inline_imm_m_4.0_f32(float addrspace(1)* %out) {
  store float -4.0, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}store_literal_imm_f32:
; CHECK: V_MOV_B32_e32 [[REG:v[0-9]+]], 0x45800000
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @store_literal_imm_f32(float addrspace(1)* %out) {
  store float 4096.0, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}add_inline_imm_0.0_f32
; CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]]
; CHECK: V_ADD_F32_e64 [[REG:v[0-9]+]], 0.0, [[VAL]]{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @add_inline_imm_0.0_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 0.0
  store float %y, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}add_inline_imm_0.5_f32
; CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]]
; CHECK: V_ADD_F32_e64 [[REG:v[0-9]+]], 0.5, [[VAL]]{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @add_inline_imm_0.5_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 0.5
  store float %y, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}add_inline_imm_neg_0.5_f32
; CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]]
; CHECK: V_ADD_F32_e64 [[REG:v[0-9]+]], -0.5, [[VAL]]{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @add_inline_imm_neg_0.5_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, -0.5
  store float %y, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}add_inline_imm_1.0_f32
; CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]]
; CHECK: V_ADD_F32_e64 [[REG:v[0-9]+]], 1.0, [[VAL]]{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @add_inline_imm_1.0_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 1.0
  store float %y, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}add_inline_imm_neg_1.0_f32
; CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]]
; CHECK: V_ADD_F32_e64 [[REG:v[0-9]+]], -1.0, [[VAL]]{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @add_inline_imm_neg_1.0_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, -1.0
  store float %y, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}add_inline_imm_2.0_f32
; CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]]
; CHECK: V_ADD_F32_e64 [[REG:v[0-9]+]], 2.0, [[VAL]]{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @add_inline_imm_2.0_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 2.0
  store float %y, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}add_inline_imm_neg_2.0_f32
; CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]]
; CHECK: V_ADD_F32_e64 [[REG:v[0-9]+]], -2.0, [[VAL]]{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @add_inline_imm_neg_2.0_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, -2.0
  store float %y, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}add_inline_imm_4.0_f32
; CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]]
; CHECK: V_ADD_F32_e64 [[REG:v[0-9]+]], 4.0, [[VAL]]{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @add_inline_imm_4.0_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 4.0
  store float %y, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}add_inline_imm_neg_4.0_f32
; CHECK: S_LOAD_DWORD [[VAL:s[0-9]+]]
; CHECK: V_ADD_F32_e64 [[REG:v[0-9]+]], -4.0, [[VAL]]{{$}}
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @add_inline_imm_neg_4.0_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, -4.0
  store float %y, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @commute_add_inline_imm_0.5_f32
; CHECK: BUFFER_LOAD_DWORD [[VAL:v[0-9]+]]
; CHECK: V_ADD_F32_e32 [[REG:v[0-9]+]], 0.5, [[VAL]]
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @commute_add_inline_imm_0.5_f32(float addrspace(1)* %out, float addrspace(1)* %in) {
  %x = load float addrspace(1)* %in
  %y = fadd float %x, 0.5
  store float %y, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @commute_add_literal_f32
; CHECK: BUFFER_LOAD_DWORD [[VAL:v[0-9]+]]
; CHECK: V_ADD_F32_e32 [[REG:v[0-9]+]], 0x44800000, [[VAL]]
; CHECK-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @commute_add_literal_f32(float addrspace(1)* %out, float addrspace(1)* %in) {
  %x = load float addrspace(1)* %in
  %y = fadd float %x, 1024.0
  store float %y, float addrspace(1)* %out
  ret void
}

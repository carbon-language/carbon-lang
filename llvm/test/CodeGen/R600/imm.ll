; RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck %s

; Use a 64-bit value with lo bits that can be represented as an inline constant
; CHECK: @i64_imm_inline_lo
; CHECK: S_MOV_B32 [[LO:s[0-9]+]], 5
; CHECK: V_MOV_B32_e32 v[[LO_VGPR:[0-9]+]], [[LO]]
; CHECK: BUFFER_STORE_DWORDX2 v{{\[}}[[LO_VGPR]]:
define void @i64_imm_inline_lo(i64 addrspace(1) *%out) {
entry:
  store i64 1311768464867721221, i64 addrspace(1) *%out ; 0x1234567800000005
  ret void
}

; Use a 64-bit value with hi bits that can be represented as an inline constant
; CHECK: @i64_imm_inline_hi
; CHECK: S_MOV_B32 [[HI:s[0-9]+]], 5
; CHECK: V_MOV_B32_e32 v[[HI_VGPR:[0-9]+]], [[HI]]
; CHECK: BUFFER_STORE_DWORDX2 v{{\[[0-9]+:}}[[HI_VGPR]]
define void @i64_imm_inline_hi(i64 addrspace(1) *%out) {
entry:
  store i64 21780256376, i64 addrspace(1) *%out ; 0x0000000512345678
  ret void
}

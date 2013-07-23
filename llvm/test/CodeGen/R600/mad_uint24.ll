; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=EG-CHECK
; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck %s --check-prefix=EG-CHECK
; RUN: llc < %s -march=r600 -mcpu=SI | FileCheck %s --check-prefix=SI-CHECK

; EG-CHECK: @u32_mad24
; EG-CHECK: MULADD_UINT24 {{[* ]*}}T{{[0-9]\.[XYZW]}}, KC0[2].Z, KC0[2].W, KC0[3].X
; SI-CHECK: @u32_mad24
; SI-CHECK: V_MAD_U32_U24

define void @u32_mad24(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) {
entry:
  %0 = shl i32 %a, 8
  %a_24 = lshr i32 %0, 8
  %1 = shl i32 %b, 8
  %b_24 = lshr i32 %1, 8
  %2 = mul i32 %a_24, %b_24
  %3 = add i32 %2, %c
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; EG-CHECK: @i16_mad24
; EG-CHECK-DAG: VTX_READ_16 [[A:T[0-9]\.X]], T{{[0-9]}}.X, 40
; EG-CHECK-DAG: VTX_READ_16 [[B:T[0-9]\.X]], T{{[0-9]}}.X, 44
; EG-CHECK-DAG: VTX_READ_16 [[C:T[0-9]\.X]], T{{[0-9]}}.X, 48
; The order of A and B does not matter.
; EG-CHECK: MULADD_UINT24 {{[* ]*}}T{{[0-9]}}.[[MAD_CHAN:[XYZW]]], [[A]], [[B]], [[C]]
; The result must be sign-extended
; EG-CHECK: LSHL {{[* ]*}}T{{[0-9]}}.[[LSHL_CHAN:[XYZW]]], PV.[[MAD_CHAN]], literal.x
; EG-CHECK: 16
; EG-CHECK: ASHR {{[* ]*}}T{{[0-9]\.[XYZW]}}, PV.[[LSHL_CHAN]], literal.x
; EG-CHECK: 16
; SI-CHECK: @i16_mad24
; SI-CHECK: V_MAD_U32_U24 [[MAD:VGPR[0-9]]], {{[SV]GPR[0-9], [SV]GPR[0-9]}}
; SI-CHECK: V_LSHLREV_B32_e32 [[LSHL:VGPR[0-9]]], 16, [[MAD]]
; SI-CHECK: V_ASHRREV_I32_e32 VGPR{{[0-9]}}, 16, [[LSHL]]

define void @i16_mad24(i32 addrspace(1)* %out, i16 %a, i16 %b, i16 %c) {
entry:
  %0 = mul i16 %a, %b
  %1 = add i16 %0, %c
  %2 = sext i16 %1 to i32
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; EG-CHECK: @i8_mad24
; EG-CHECK-DAG: VTX_READ_8 [[A:T[0-9]\.X]], T{{[0-9]}}.X, 40
; EG-CHECK-DAG: VTX_READ_8 [[B:T[0-9]\.X]], T{{[0-9]}}.X, 44
; EG-CHECK-DAG: VTX_READ_8 [[C:T[0-9]\.X]], T{{[0-9]}}.X, 48
; The order of A and B does not matter.
; EG-CHECK: MULADD_UINT24 {{[* ]*}}T{{[0-9]}}.[[MAD_CHAN:[XYZW]]], [[A]], [[B]], [[C]]
; The result must be sign-extended
; EG-CHECK: LSHL {{[* ]*}}T{{[0-9]}}.[[LSHL_CHAN:[XYZW]]], PV.[[MAD_CHAN]], literal.x
; EG-CHECK: 24
; EG-CHECK: ASHR {{[* ]*}}T{{[0-9]\.[XYZW]}}, PV.[[LSHL_CHAN]], literal.x
; EG-CHECK: 24
; SI-CHECK: @i8_mad24
; SI-CHECK: V_MAD_U32_U24 [[MUL:VGPR[0-9]]], {{[SV]GPR[0-9], [SV]GPR[0-9]}}
; SI-CHECK: V_LSHLREV_B32_e32 [[LSHL:VGPR[0-9]]], 24, [[MUL]]
; SI-CHECK: V_ASHRREV_I32_e32 VGPR{{[0-9]}}, 24, [[LSHL]]

define void @i8_mad24(i32 addrspace(1)* %out, i8 %a, i8 %b, i8 %c) {
entry:
  %0 = mul i8 %a, %b
  %1 = add i8 %0, %c
  %2 = sext i8 %1 to i32
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=EG-CHECK
; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck %s --check-prefix=EG-CHECK
; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=SI-CHECK

; EG-CHECK-LABEL: @u32_mul24
; EG-CHECK: MUL_UINT24 {{[* ]*}}T{{[0-9]\.[XYZW]}}, KC0[2].Z, KC0[2].W
; SI-CHECK-LABEL: @u32_mul24
; SI-CHECK: V_MUL_U32_U24

define void @u32_mul24(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = shl i32 %a, 8
  %a_24 = lshr i32 %0, 8
  %1 = shl i32 %b, 8
  %b_24 = lshr i32 %1, 8
  %2 = mul i32 %a_24, %b_24
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; EG-CHECK-LABEL: @i16_mul24
; EG-CHECK-DAG: VTX_READ_16 [[A:T[0-9]\.X]], T{{[0-9]}}.X, 40
; EG-CHECK-DAG: VTX_READ_16 [[B:T[0-9]\.X]], T{{[0-9]}}.X, 44
; The order of A and B does not matter.
; EG-CHECK: MUL_UINT24 {{[* ]*}}T{{[0-9]}}.[[MUL_CHAN:[XYZW]]], [[A]], [[B]]
; The result must be sign-extended
; EG-CHECK: BFE_INT {{[* ]*}}T{{[0-9]}}.{{[XYZW]}}, PV.[[MUL_CHAN]], 0.0, literal.x
; EG-CHECK: 16
; SI-CHECK-LABEL: @i16_mul24
; SI-CHECK: V_MUL_U32_U24_e{{(32|64)}} [[MUL:v[0-9]]], {{[sv][0-9], [sv][0-9]}}
; SI-CHECK: V_BFE_I32 v{{[0-9]}}, [[MUL]], 0, 16,
define void @i16_mul24(i32 addrspace(1)* %out, i16 %a, i16 %b) {
entry:
  %0 = mul i16 %a, %b
  %1 = sext i16 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; EG-CHECK-LABEL: @i8_mul24
; EG-CHECK-DAG: VTX_READ_8 [[A:T[0-9]\.X]], T{{[0-9]}}.X, 40
; EG-CHECK-DAG: VTX_READ_8 [[B:T[0-9]\.X]], T{{[0-9]}}.X, 44
; The order of A and B does not matter.
; EG-CHECK: MUL_UINT24 {{[* ]*}}T{{[0-9]}}.[[MUL_CHAN:[XYZW]]], [[A]], [[B]]
; The result must be sign-extended
; EG-CHECK: BFE_INT {{[* ]*}}T{{[0-9]}}.{{[XYZW]}}, PV.[[MUL_CHAN]], 0.0, literal.x
; SI-CHECK-LABEL: @i8_mul24
; SI-CHECK: V_MUL_U32_U24_e{{(32|64)}} [[MUL:v[0-9]]], {{[sv][0-9], [sv][0-9]}}
; SI-CHECK: V_BFE_I32 v{{[0-9]}}, [[MUL]], 0, 8,

define void @i8_mul24(i32 addrspace(1)* %out, i8 %a, i8 %b) {
entry:
  %0 = mul i8 %a, %b
  %1 = sext i8 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

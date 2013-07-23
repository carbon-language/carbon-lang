; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=EG-CHECK
; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck %s --check-prefix=CM-CHECK
; RUN: llc < %s -march=r600 -mcpu=SI | FileCheck %s --check-prefix=SI-CHECK

; EG-CHECK: @i32_mad24
; Signed 24-bit multiply is not supported on pre-Cayman GPUs.
; EG-CHECK: MULLO_INT
; CM-CHECK: MULADD_INT24 {{[ *]*}}T{{[0-9].[XYZW]}}, KC0[2].Z, KC0[2].W, KC0[3].X
; SI-CHECK: V_MAD_I32_I24
define void @i32_mad24(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) {
entry:
  %0 = shl i32 %a, 8
  %a_24 = ashr i32 %0, 8
  %1 = shl i32 %b, 8
  %b_24 = ashr i32 %1, 8
  %2 = mul i32 %a_24, %b_24
  %3 = add i32 %2, %c
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

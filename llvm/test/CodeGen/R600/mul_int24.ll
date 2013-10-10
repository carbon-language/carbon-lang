; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=EG-CHECK
; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck %s --check-prefix=CM-CHECK
; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=SI-CHECK

; EG-CHECK: @i32_mul24
; Signed 24-bit multiply is not supported on pre-Cayman GPUs.
; EG-CHECK: MULLO_INT
; CM-CHECK: MUL_INT24 {{[ *]*}}T{{[0-9].[XYZW]}}, KC0[2].Z, KC0[2].W
; SI-CHECK: V_MUL_I32_I24
define void @i32_mul24(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = shl i32 %a, 8
  %a_24 = ashr i32 %0, 8
  %1 = shl i32 %b, 8
  %b_24 = ashr i32 %1, 8
  %2 = mul i32 %a_24, %b_24
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=EG --check-prefix=FUNC
; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck %s --check-prefix=EG --check-prefix=FUNC
; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=SI --check-prefix=FUNC

; FIXME: Move this test into mul_uint24.ll once i64 mul is supported.
; XFAIL: *

; Multiply with 24-bit inputs and 64-bit output
; FUNC_LABEL: @mul24_i64
; EG; MUL_UINT24
; EG: MULHI
; SI: V_MUL_U32_U24
; FIXME: SI support 24-bit mulhi
; SI: V_MUL_HI_U32
define void @mul24_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %0 = shl i64 %a, 40
  %a_24 = lshr i64 %0, 40
  %1 = shl i64 %b, 40
  %b_24 = lshr i64 %1, 40
  %2 = mul i64 %a_24, %b_24
  store i64 %2, i64 addrspace(1)* %out
  ret void
}

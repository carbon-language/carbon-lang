; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=R600-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck --check-prefix=SI-CHECK %s

; R600-CHECK-LABEL: @rotr:
; R600-CHECK: BIT_ALIGN_INT

; SI-CHECK-LABEL: @rotr:
; SI-CHECK: V_ALIGNBIT_B32
define void @rotr(i32 addrspace(1)* %in, i32 %x, i32 %y) {
entry:
  %0 = sub i32 32, %y
  %1 = shl i32 %x, %0
  %2 = lshr i32 %x, %y
  %3 = or i32 %1, %2
  store i32 %3, i32 addrspace(1)* %in
  ret void
}

; R600-CHECK-LABEL: @rotl:
; R600-CHECK: SUB_INT {{\** T[0-9]+\.[XYZW]}}, literal.x
; R600-CHECK-NEXT: 32
; R600-CHECK: BIT_ALIGN_INT {{T[0-9]+\.[XYZW]}}, KC0[2].Z, KC0[2].Z, PV.{{[XYZW]}}


; SI-CHECK-LABEL: @rotl:
; SI-CHECK: S_SUB_I32 [[SDST:s[0-9]+]], 32, {{[s][0-9]+}}
; SI-CHECK: V_MOV_B32_e32 [[VDST:v[0-9]+]], [[SDST]]
; SI-CHECK: V_ALIGNBIT_B32 {{v[0-9]+, [s][0-9]+, v[0-9]+}}, [[VDST]]
define void @rotl(i32 addrspace(1)* %in, i32 %x, i32 %y) {
entry:
  %0 = shl i32 %x, %y
  %1 = sub i32 32, %y
  %2 = lshr i32 %x, %1
  %3 = or i32 %0, %2
  store i32 %3, i32 addrspace(1)* %in
  ret void
}

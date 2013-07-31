; RUN: llc < %s -march=r600 -mcpu=redwood -o - | FileCheck --check-prefix=R600-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=SI -o - | FileCheck --check-prefix=SI-CHECK %s

; R600-CHECK: @rotr
; R600-CHECK: BIT_ALIGN_INT

; SI-CHECK: @rotr
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

; R600-CHECK: @rotl
; R600-CHECK: SUB_INT {{\** T[0-9]+\.[XYZW]}}, literal.x
; R600-CHECK-NEXT: 32
; R600-CHECK: BIT_ALIGN_INT {{\** T[0-9]+\.[XYZW]}}, KC0[2].Z, KC0[2].Z, PV.{{[XYZW]}}

; SI-CHECK: @rotl
; SI-CHECK: V_SUB_I32_e64 [[DST:VGPR[0-9]+]], 32, {{[SV]GPR[0-9]+}}
; SI-CHECK: V_ALIGNBIT_B32 {{VGPR[0-9]+, [SV]GPR[0-9]+, VGPR[0-9]+}}, [[DST]]
define void @rotl(i32 addrspace(1)* %in, i32 %x, i32 %y) {
entry:
  %0 = shl i32 %x, %y
  %1 = sub i32 32, %y
  %2 = lshr i32 %x, %1
  %3 = or i32 %0, %2
  store i32 %3, i32 addrspace(1)* %in
  ret void
}

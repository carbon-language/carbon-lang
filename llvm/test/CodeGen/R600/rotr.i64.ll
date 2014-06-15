; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: @s_rotr_i64
; SI: S_LSHR_B64
; SI: S_SUB_I32
; SI: S_LSHL_B64
; SI: S_OR_B64
define void @s_rotr_i64(i64 addrspace(1)* %in, i64 %x, i64 %y) {
entry:
  %tmp0 = sub i64 64, %y
  %tmp1 = shl i64 %x, %tmp0
  %tmp2 = lshr i64 %x, %y
  %tmp3 = or i64 %tmp1, %tmp2
  store i64 %tmp3, i64 addrspace(1)* %in
  ret void
}

; FUNC-LABEL: @v_rotr_i64
; SI: V_LSHR_B64
; SI: V_SUB_I32
; SI: V_LSHL_B64
; SI: V_OR_B32
; SI: V_OR_B32
define void @v_rotr_i64(i64 addrspace(1)* %in, i64 addrspace(1)* %xptr, i64 addrspace(1)* %yptr) {
entry:
  %x = load i64 addrspace(1)* %xptr, align 8
  %y = load i64 addrspace(1)* %yptr, align 8
  %tmp0 = sub i64 64, %y
  %tmp1 = shl i64 %x, %tmp0
  %tmp2 = lshr i64 %x, %y
  %tmp3 = or i64 %tmp1, %tmp2
  store i64 %tmp3, i64 addrspace(1)* %in
  ret void
}

; FUNC-LABEL: @s_rotr_v2i64
define void @s_rotr_v2i64(<2 x i64> addrspace(1)* %in, <2 x i64> %x, <2 x i64> %y) {
entry:
  %tmp0 = sub <2 x i64> <i64 64, i64 64>, %y
  %tmp1 = shl <2 x i64> %x, %tmp0
  %tmp2 = lshr <2 x i64> %x, %y
  %tmp3 = or <2 x i64> %tmp1, %tmp2
  store <2 x i64> %tmp3, <2 x i64> addrspace(1)* %in
  ret void
}

; FUNC-LABEL: @v_rotr_v2i64
define void @v_rotr_v2i64(<2 x i64> addrspace(1)* %in, <2 x i64> addrspace(1)* %xptr, <2 x i64> addrspace(1)* %yptr) {
entry:
  %x = load <2 x i64> addrspace(1)* %xptr, align 8
  %y = load <2 x i64> addrspace(1)* %yptr, align 8
  %tmp0 = sub <2 x i64> <i64 64, i64 64>, %y
  %tmp1 = shl <2 x i64> %x, %tmp0
  %tmp2 = lshr <2 x i64> %x, %y
  %tmp3 = or <2 x i64> %tmp1, %tmp2
  store <2 x i64> %tmp3, <2 x i64> addrspace(1)* %in
  ret void
}

; RUN: llc < %s -debug-only=isel -march=r600 -mcpu=redwood -o - 2>&1 | FileCheck %s

; CHECK: rotr
; CHECK: @rotr
; CHECK: BIT_ALIGN_INT
define void @rotr(i32 addrspace(1)* %in, i32 %x, i32 %y) {
entry:
  %0 = sub i32 32, %y
  %1 = shl i32 %x, %0
  %2 = lshr i32 %x, %y
  %3 = or i32 %1, %2
  store i32 %3, i32 addrspace(1)* %in
  ret void
}

; CHECK: rotr
; CHECK: @rotl
; CHECK: SUB_INT {{\** T[0-9]+\.[XYZW]}}, literal.x
; CHECK-NEXT: 32
; CHECK: BIT_ALIGN_INT {{\** T[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW], PV.[xyzw]}}
define void @rotl(i32 addrspace(1)* %in, i32 %x, i32 %y) {
entry:
  %0 = shl i32 %x, %y
  %1 = sub i32 32, %y
  %2 = lshr i32 %x, %1
  %3 = or i32 %0, %2
  store i32 %3, i32 addrspace(1)* %in
  ret void
}

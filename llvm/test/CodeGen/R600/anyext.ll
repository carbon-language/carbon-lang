; RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: @anyext_i1_i32
; CHECK: V_CNDMASK_B32_e64
define void @anyext_i1_i32(i32 addrspace(1)* %out, i32 %cond) {
entry:
  %0 = icmp eq i32 %cond, 0
  %1 = zext i1 %0 to i8
  %2 = xor i8 %1, -1
  %3 = and i8 %2, 1
  %4 = zext i8 %3 to i32
  store i32 %4, i32 addrspace(1)* %out
  ret void
}

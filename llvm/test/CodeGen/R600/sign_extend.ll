
; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s

; CHECK: V_ASHR
define void @test(i64 addrspace(1)* %out, i32 %a, i32 %b, i32 %c)  {
entry:
  %0 = mul i32 %a, %b
  %1 = add i32 %0, %c
  %2 = sext i32 %1 to i64
  store i64 %2, i64 addrspace(1)* %out
  ret void
}

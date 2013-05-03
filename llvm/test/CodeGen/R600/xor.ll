; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: @xor_v4i32
; CHECK: XOR_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: XOR_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: XOR_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: XOR_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @xor_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b) {
  %result = xor <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: @or_v4i32
; CHECK: OR_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: OR_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: OR_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: OR_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @or_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b) {
  %result = or <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: @sint_to_fp_v4i32
; CHECK: INT_TO_FLT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: INT_TO_FLT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: INT_TO_FLT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: INT_TO_FLT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @sint_to_fp_v4i32(<4 x float> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %value = load <4 x i32> addrspace(1) * %in
  %result = sitofp <4 x i32> %value to <4 x float>
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}

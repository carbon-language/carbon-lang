; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: @fp_to_uint_v2i32
; CHECK: FLT_TO_UINT * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; CHECK: FLT_TO_UINT * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}

define void @fp_to_uint_v2i32(<2 x i32> addrspace(1)* %out, <2 x float> %in) {
  %result = fptoui <2 x float> %in to <2 x i32>
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; CHECK: @fp_to_uint_v4i32
; CHECK: FLT_TO_UINT * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; CHECK: FLT_TO_UINT * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; CHECK: FLT_TO_UINT * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; CHECK: FLT_TO_UINT * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}

define void @fp_to_uint_v4i32(<4 x i32> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
  %value = load <4 x float> addrspace(1) * %in
  %result = fptoui <4 x float> %value to <4 x i32>
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

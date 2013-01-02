; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: FLT_TO_INT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: FLT_TO_INT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: FLT_TO_INT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: FLT_TO_INT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @fp_to_sint(<4 x i32> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
  %value = load <4 x float> addrspace(1) * %in
  %result = fptosi <4 x float> %value to <4 x i32>
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; CHECK: FLT_TO_UINT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: FLT_TO_UINT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: FLT_TO_UINT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: FLT_TO_UINT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @fp_to_uint(<4 x i32> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
  %value = load <4 x float> addrspace(1) * %in
  %result = fptoui <4 x float> %value to <4 x i32>
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; CHECK: INT_TO_FLT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: INT_TO_FLT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: INT_TO_FLT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: INT_TO_FLT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @sint_to_fp(<4 x float> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %value = load <4 x i32> addrspace(1) * %in
  %result = sitofp <4 x i32> %value to <4 x float>
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}

; CHECK: UINT_TO_FLT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: UINT_TO_FLT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: UINT_TO_FLT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: UINT_TO_FLT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @uint_to_fp(<4 x float> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %value = load <4 x i32> addrspace(1) * %in
  %result = uitofp <4 x i32> %value to <4 x float>
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}

;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG-CHECK %s
;RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI-CHECK %s

;EG-CHECK: @shl_v2i32
;EG-CHECK: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI-CHECK: @shl_v2i32
;SI-CHECK: V_LSHL_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: V_LSHL_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @shl_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32> addrspace(1) * %in
  %b = load <2 x i32> addrspace(1) * %b_ptr
  %result = shl <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

;EG-CHECK: @shl_v4i32
;EG-CHECK: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: LSHL {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI-CHECK: @shl_v4i32
;SI-CHECK: V_LSHL_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: V_LSHL_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: V_LSHL_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: V_LSHL_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @shl_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32> addrspace(1) * %in
  %b = load <4 x i32> addrspace(1) * %b_ptr
  %result = shl <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; XXX: Add SI test for i64 shl once i64 stores and i64 function arguments are
; supported.

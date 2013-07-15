;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG-CHECK %s
;RUN: llc < %s -march=r600 -mcpu=verde | FileCheck --check-prefix=SI-CHECK %s

;EG-CHECK: @ashr_v2i32
;EG-CHECK: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI-CHECK: @ashr_v2i32
;SI-CHECK: V_ASHR_I32_e32 VGPR{{[0-9]+, VGPR[0-9]+, VGPR[0-9]+}}
;SI-CHECK: V_ASHR_I32_e32 VGPR{{[0-9]+, VGPR[0-9]+, VGPR[0-9]+}}

define void @ashr_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32> addrspace(1) * %in
  %b = load <2 x i32> addrspace(1) * %b_ptr
  %result = ashr <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

;EG-CHECK: @ashr_v4i32
;EG-CHECK: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: ASHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI-CHECK: @ashr_v4i32
;SI-CHECK: V_ASHR_I32_e32 VGPR{{[0-9]+, VGPR[0-9]+, VGPR[0-9]+}}
;SI-CHECK: V_ASHR_I32_e32 VGPR{{[0-9]+, VGPR[0-9]+, VGPR[0-9]+}}
;SI-CHECK: V_ASHR_I32_e32 VGPR{{[0-9]+, VGPR[0-9]+, VGPR[0-9]+}}
;SI-CHECK: V_ASHR_I32_e32 VGPR{{[0-9]+, VGPR[0-9]+, VGPR[0-9]+}}

define void @ashr_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32> addrspace(1) * %in
  %b = load <4 x i32> addrspace(1) * %b_ptr
  %result = ashr <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

;EG-CHECK: @ashr_i64
;EG-CHECK: ASHR

;SI-CHECK: @ashr_i64
;SI-CHECK: V_ASHR_I64
define void @ashr_i64(i64 addrspace(1)* %out, i32 %in) {
entry:
  %0 = sext i32 %in to i64
  %1 = ashr i64 %0, 8
  store i64 %1, i64 addrspace(1)* %out
  ret void
}


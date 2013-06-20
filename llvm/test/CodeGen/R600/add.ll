; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=verde | FileCheck --check-prefix=SI-CHECK %s

;EG-CHECK: @test2
;EG-CHECK: ADD_INT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: ADD_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], literal\.[xyzw]}}

;SI-CHECK: @test2
;SI-CHECK: V_ADD_I32_e32 VGPR{{[0-9]+, VGPR[0-9]+, VGPR[0-9]+}}
;SI-CHECK: V_ADD_I32_e32 VGPR{{[0-9]+, VGPR[0-9]+, VGPR[0-9]+}}

define void @test2(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32> addrspace(1) * %in
  %b = load <2 x i32> addrspace(1) * %b_ptr
  %result = add <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

;EG-CHECK: @test4
;EG-CHECK: ADD_INT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: ADD_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: ADD_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: ADD_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI-CHECK: @test4
;SI-CHECK: V_ADD_I32_e32 VGPR{{[0-9]+, VGPR[0-9]+, VGPR[0-9]+}}
;SI-CHECK: V_ADD_I32_e32 VGPR{{[0-9]+, VGPR[0-9]+, VGPR[0-9]+}}
;SI-CHECK: V_ADD_I32_e32 VGPR{{[0-9]+, VGPR[0-9]+, VGPR[0-9]+}}
;SI-CHECK: V_ADD_I32_e32 VGPR{{[0-9]+, VGPR[0-9]+, VGPR[0-9]+}}

define void @test4(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32> addrspace(1) * %in
  %b = load <4 x i32> addrspace(1) * %b_ptr
  %result = add <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG-CHECK --check-prefix=FUNC %s
; RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI-CHECK --check-prefix=FUNC %s

;FUNC-LABEL: @test1:
;EG-CHECK: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI-CHECK: V_ADD_I32_e32 [[REG:v[0-9]+]], {{v[0-9]+, v[0-9]+}}
;SI-CHECK-NOT: [[REG]]
;SI-CHECK: BUFFER_STORE_DWORD [[REG]],
define void @test1(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %b_ptr = getelementptr i32 addrspace(1)* %in, i32 1
  %a = load i32 addrspace(1)* %in
  %b = load i32 addrspace(1)* %b_ptr
  %result = add i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

;FUNC-LABEL: @test2:
;EG-CHECK: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI-CHECK: V_ADD_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: V_ADD_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @test2(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32> addrspace(1)* %in
  %b = load <2 x i32> addrspace(1)* %b_ptr
  %result = add <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

;FUNC-LABEL: @test4:
;EG-CHECK: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI-CHECK: V_ADD_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: V_ADD_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: V_ADD_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: V_ADD_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @test4(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32> addrspace(1)* %in
  %b = load <4 x i32> addrspace(1)* %b_ptr
  %result = add <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @test8
; EG-CHECK: ADD_INT
; EG-CHECK: ADD_INT
; EG-CHECK: ADD_INT
; EG-CHECK: ADD_INT
; EG-CHECK: ADD_INT
; EG-CHECK: ADD_INT
; EG-CHECK: ADD_INT
; EG-CHECK: ADD_INT
; SI-CHECK: S_ADD_I32
; SI-CHECK: S_ADD_I32
; SI-CHECK: S_ADD_I32
; SI-CHECK: S_ADD_I32
; SI-CHECK: S_ADD_I32
; SI-CHECK: S_ADD_I32
; SI-CHECK: S_ADD_I32
; SI-CHECK: S_ADD_I32
define void @test8(<8 x i32> addrspace(1)* %out, <8 x i32> %a, <8 x i32> %b) {
entry:
  %0 = add <8 x i32> %a, %b
  store <8 x i32> %0, <8 x i32> addrspace(1)* %out
  ret void
}

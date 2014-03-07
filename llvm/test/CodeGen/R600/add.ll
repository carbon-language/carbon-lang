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

; FUNC-LABEL: @test16
; EG-CHECK: ADD_INT
; EG-CHECK: ADD_INT
; EG-CHECK: ADD_INT
; EG-CHECK: ADD_INT
; EG-CHECK: ADD_INT
; EG-CHECK: ADD_INT
; EG-CHECK: ADD_INT
; EG-CHECK: ADD_INT
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
; SI-CHECK: S_ADD_I32
; SI-CHECK: S_ADD_I32
; SI-CHECK: S_ADD_I32
; SI-CHECK: S_ADD_I32
; SI-CHECK: S_ADD_I32
; SI-CHECK: S_ADD_I32
; SI-CHECK: S_ADD_I32
; SI-CHECK: S_ADD_I32
define void @test16(<16 x i32> addrspace(1)* %out, <16 x i32> %a, <16 x i32> %b) {
entry:
  %0 = add <16 x i32> %a, %b
  store <16 x i32> %0, <16 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @add64
; SI-CHECK: S_ADD_I32
; SI-CHECK: S_ADDC_U32
define void @add64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %0 = add i64 %a, %b
  store i64 %0, i64 addrspace(1)* %out
  ret void
}

; The V_ADDC_U32 and V_ADD_I32 instruction can't read SGPRs, because they
; use VCC.  The test is designed so that %a will be stored in an SGPR and
; %0 will be stored in a VGPR, so the comiler will be forced to copy %a
; to a VGPR before doing the add.

; FUNC-LABEL: @add64_sgpr_vgpr
; SI-CHECK-NOT: V_ADDC_U32_e32 s
define void @add64_sgpr_vgpr(i64 addrspace(1)* %out, i64 %a, i64 addrspace(1)* %in) {
entry:
  %0 = load i64 addrspace(1)* %in
  %1 = add i64 %a, %0
  store i64 %1, i64 addrspace(1)* %out
  ret void
}

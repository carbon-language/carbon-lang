; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI-CHECK %s

; mul24 and mad24 are affected

;EG-CHECK: @test2
;EG-CHECK: MULLO_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: MULLO_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI-CHECK: @test2
;SI-CHECK: V_MUL_LO_I32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: V_MUL_LO_I32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @test2(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32> addrspace(1) * %in
  %b = load <2 x i32> addrspace(1) * %b_ptr
  %result = mul <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

;EG-CHECK: @test4
;EG-CHECK: MULLO_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: MULLO_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: MULLO_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: MULLO_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI-CHECK: @test4
;SI-CHECK: V_MUL_LO_I32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: V_MUL_LO_I32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: V_MUL_LO_I32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: V_MUL_LO_I32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @test4(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32> addrspace(1) * %in
  %b = load <4 x i32> addrspace(1) * %b_ptr
  %result = mul <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; SI-CHECK-LABEL: @trunc_i64_mul_to_i32
; SI-CHECK: S_LOAD_DWORD
; SI-CHECK: S_LOAD_DWORD
; SI-CHECK: V_MUL_LO_I32
; SI-CHECK: BUFFER_STORE_DWORD
define void @trunc_i64_mul_to_i32(i32 addrspace(1)* %out, i64 %a, i64 %b) {
  %mul = mul i64 %b, %a
  %trunc = trunc i64 %mul to i32
  store i32 %trunc, i32 addrspace(1)* %out, align 8
  ret void
}

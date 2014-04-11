; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG %s --check-prefix=FUNC
; RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI --check-prefix=FUNC %s

; mul24 and mad24 are affected

;FUNC-LABEL: @test2
;EG: MULLO_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: MULLO_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI: V_MUL_LO_I32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: V_MUL_LO_I32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @test2(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32> addrspace(1) * %in
  %b = load <2 x i32> addrspace(1) * %b_ptr
  %result = mul <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

;FUNC-LABEL: @test4
;EG: MULLO_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: MULLO_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: MULLO_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: MULLO_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI: V_MUL_LO_I32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: V_MUL_LO_I32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: V_MUL_LO_I32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: V_MUL_LO_I32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

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

; This 64-bit multiply should just use MUL_HI and MUL_LO, since the top
; 32-bits of both arguments are sign bits.
; FUNC-LABEL: @mul64_sext_c
; EG-DAG: MULLO_INT
; EG-DAG: MULHI_INT
; SI-DAG: V_MUL_LO_I32
; SI-DAG: V_MUL_HI_I32
define void @mul64_sext_c(i64 addrspace(1)* %out, i32 %in) {
entry:
  %0 = sext i32 %in to i64
  %1 = mul i64 %0, 80
  store i64 %1, i64 addrspace(1)* %out
  ret void
}

; A standard 64-bit multiply.  The expansion should be around 6 instructions.
; It would be difficult to match the expansion correctly without writing
; a really complicated list of FileCheck expressions.  I don't want
; to confuse people who may 'break' this test with a correct optimization,
; so this test just uses FUNC-LABEL to make sure the compiler does not
; crash with a 'failed to select' error.
; FUNC-LABEL: @mul64
define void @mul64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %0 = mul i64 %a, %b
  store i64 %0, i64 addrspace(1)* %out
  ret void
}

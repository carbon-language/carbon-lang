; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: @test2
; EG: AND_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: AND_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; SI: V_AND_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: V_AND_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @test2(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32> addrspace(1) * %in
  %b = load <2 x i32> addrspace(1) * %b_ptr
  %result = and <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @test4
; EG: AND_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: AND_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: AND_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: AND_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; SI: V_AND_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: V_AND_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: V_AND_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: V_AND_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @test4(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32> addrspace(1) * %in
  %b = load <4 x i32> addrspace(1) * %b_ptr
  %result = and <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @s_and_i32
; SI: S_AND_B32
define void @s_and_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) {
  %and = and i32 %a, %b
  store i32 %and, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @s_and_constant_i32
; SI: S_AND_B32 s{{[0-9]+}}, s{{[0-9]+}}, 0x12d687
define void @s_and_constant_i32(i32 addrspace(1)* %out, i32 %a) {
  %and = and i32 %a, 1234567
  store i32 %and, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_and_i32
; SI: V_AND_B32
define void @v_and_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) {
  %a = load i32 addrspace(1)* %aptr, align 4
  %b = load i32 addrspace(1)* %bptr, align 4
  %and = and i32 %a, %b
  store i32 %and, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_and_constant_i32
; SI: V_AND_B32
define void @v_and_constant_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr) {
  %a = load i32 addrspace(1)* %aptr, align 4
  %and = and i32 %a, 1234567
  store i32 %and, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @s_and_i64
; SI: S_AND_B64
define void @s_and_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
  %and = and i64 %a, %b
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @s_and_constant_i64
; SI: S_AND_B64
define void @s_and_constant_i64(i64 addrspace(1)* %out, i64 %a) {
  %and = and i64 %a, 281474976710655
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @v_and_i64
; SI: V_AND_B32
; SI: V_AND_B32
define void @v_and_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) {
  %a = load i64 addrspace(1)* %aptr, align 8
  %b = load i64 addrspace(1)* %bptr, align 8
  %and = and i64 %a, %b
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @v_and_constant_i64
; SI: V_AND_B32
; SI: V_AND_B32
define void @v_and_constant_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr) {
  %a = load i64 addrspace(1)* %aptr, align 8
  %and = and i64 %a, 1234567
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

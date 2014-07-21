;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG-CHECK %s
;RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI-CHECK %s

;EG-CHECK: @xor_v2i32
;EG-CHECK: XOR_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: XOR_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI-CHECK: @xor_v2i32
;SI-CHECK: V_XOR_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: V_XOR_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}


define void @xor_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in0, <2 x i32> addrspace(1)* %in1) {
  %a = load <2 x i32> addrspace(1) * %in0
  %b = load <2 x i32> addrspace(1) * %in1
  %result = xor <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

;EG-CHECK: @xor_v4i32
;EG-CHECK: XOR_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: XOR_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: XOR_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG-CHECK: XOR_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI-CHECK: @xor_v4i32
;SI-CHECK: V_XOR_B32_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: V_XOR_B32_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: V_XOR_B32_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}
;SI-CHECK: V_XOR_B32_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}

define void @xor_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in0, <4 x i32> addrspace(1)* %in1) {
  %a = load <4 x i32> addrspace(1) * %in0
  %b = load <4 x i32> addrspace(1) * %in1
  %result = xor <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

;EG-CHECK: @xor_i1
;EG-CHECK: XOR_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], PS}}

;SI-CHECK: @xor_i1
;SI-CHECK: V_XOR_B32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

define void @xor_i1(float addrspace(1)* %out, float addrspace(1)* %in0, float addrspace(1)* %in1) {
  %a = load float addrspace(1) * %in0
  %b = load float addrspace(1) * %in1
  %acmp = fcmp oge float %a, 0.000000e+00
  %bcmp = fcmp oge float %b, 0.000000e+00
  %xor = xor i1 %acmp, %bcmp
  %result = select i1 %xor, float %a, float %b
  store float %result, float addrspace(1)* %out
  ret void
}

; SI-CHECK-LABEL: @vector_xor_i32
; SI-CHECK: V_XOR_B32_e32
define void @vector_xor_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in0, i32 addrspace(1)* %in1) {
  %a = load i32 addrspace(1)* %in0
  %b = load i32 addrspace(1)* %in1
  %result = xor i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; SI-CHECK-LABEL: @scalar_xor_i32
; SI-CHECK: S_XOR_B32
define void @scalar_xor_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) {
  %result = xor i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; SI-CHECK-LABEL: @scalar_not_i32
; SI-CHECK: S_NOT_B32
define void @scalar_not_i32(i32 addrspace(1)* %out, i32 %a) {
  %result = xor i32 %a, -1
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; SI-CHECK-LABEL: @vector_not_i32
; SI-CHECK: V_NOT_B32
define void @vector_not_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in0, i32 addrspace(1)* %in1) {
  %a = load i32 addrspace(1)* %in0
  %b = load i32 addrspace(1)* %in1
  %result = xor i32 %a, -1
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; SI-CHECK-LABEL: @vector_xor_i64
; SI-CHECK: V_XOR_B32_e32
; SI-CHECK: V_XOR_B32_e32
; SI-CHECK: S_ENDPGM
define void @vector_xor_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in0, i64 addrspace(1)* %in1) {
  %a = load i64 addrspace(1)* %in0
  %b = load i64 addrspace(1)* %in1
  %result = xor i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; SI-CHECK-LABEL: @scalar_xor_i64
; SI-CHECK: S_XOR_B64
; SI-CHECK: S_ENDPGM
define void @scalar_xor_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
  %result = xor i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; SI-CHECK-LABEL: @scalar_not_i64
; SI-CHECK: S_NOT_B64
define void @scalar_not_i64(i64 addrspace(1)* %out, i64 %a) {
  %result = xor i64 %a, -1
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; SI-CHECK-LABEL: @vector_not_i64
; SI-CHECK: V_NOT_B32
; SI-CHECK: V_NOT_B32
define void @vector_not_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in0, i64 addrspace(1)* %in1) {
  %a = load i64 addrspace(1)* %in0
  %b = load i64 addrspace(1)* %in1
  %result = xor i64 %a, -1
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; Test that we have a pattern to match xor inside a branch.
; Note that in the future the backend may be smart enough to
; use an SALU instruction for this.

; SI-CHECK-LABEL: @xor_cf
; SI-CHECK: V_XOR
; SI-CHECK: V_XOR
define void @xor_cf(i64 addrspace(1)* %out, i64 addrspace(1)* %in, i64 %a, i64 %b) {
entry:
  %0 = icmp eq i64 %a, 0
  br i1 %0, label %if, label %else

if:
  %1 = xor i64 %a, %b
  br label %endif

else:
  %2 = load i64 addrspace(1)* %in
  br label %endif

endif:
  %3 = phi i64 [%1, %if], [%2, %else]
  store i64 %3, i64 addrspace(1)* %out
  ret void
}

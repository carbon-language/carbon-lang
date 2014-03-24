;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG %s
;RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI %s

; EG-LABEL: @or_v2i32
; EG: OR_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: OR_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; SI-LABEL: @or_v2i32
; SI: V_OR_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: V_OR_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @or_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32> addrspace(1) * %in
  %b = load <2 x i32> addrspace(1) * %b_ptr
  %result = or <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; EG-LABEL: @or_v4i32
; EG: OR_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: OR_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: OR_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: OR_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; SI-LABEL: @or_v4i32
; SI: V_OR_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: V_OR_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: V_OR_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: V_OR_B32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @or_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32> addrspace(1) * %in
  %b = load <4 x i32> addrspace(1) * %b_ptr
  %result = or <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; SI-LABEL: @scalar_or_i32
; SI: S_OR_B32
define void @scalar_or_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) {
  %or = or i32 %a, %b
  store i32 %or, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: @vector_or_i32
; SI: V_OR_B32_e32 v{{[0-9]}}
define void @vector_or_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %a, i32 %b) {
  %loada = load i32 addrspace(1)* %a
  %or = or i32 %loada, %b
  store i32 %or, i32 addrspace(1)* %out
  ret void
}

; EG-LABEL: @scalar_or_i64
; EG-DAG: OR_INT * T{{[0-9]\.[XYZW]}}, KC0[2].W, KC0[3].Y
; EG-DAG: OR_INT * T{{[0-9]\.[XYZW]}}, KC0[3].X, KC0[3].Z
; SI-LABEL: @scalar_or_i64
; SI: S_OR_B64
define void @scalar_or_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
  %or = or i64 %a, %b
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; SI-LABEL: @vector_or_i64
; SI: V_OR_B32_e32 v{{[0-9]}}
; SI: V_OR_B32_e32 v{{[0-9]}}
define void @vector_or_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %a, i64 addrspace(1)* %b) {
  %loada = load i64 addrspace(1)* %a, align 8
  %loadb = load i64 addrspace(1)* %a, align 8
  %or = or i64 %loada, %loadb
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; SI-LABEL: @scalar_vector_or_i64
; SI: V_OR_B32_e32 v{{[0-9]}}
; SI: V_OR_B32_e32 v{{[0-9]}}
define void @scalar_vector_or_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %a, i64 %b) {
  %loada = load i64 addrspace(1)* %a
  %or = or i64 %loada, %b
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; SI-LABEL: @vector_or_i64_loadimm
; SI-DAG: S_MOV_B32 [[LO_S_IMM:s[0-9]+]], -545810305
; SI-DAG: S_MOV_B32 [[HI_S_IMM:s[0-9]+]], 5231
; SI-DAG: BUFFER_LOAD_DWORDX2 v{{\[}}[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]{{\]}},
; SI-DAG: V_OR_B32_e32 {{v[0-9]+}}, [[LO_S_IMM]], v[[LO_VREG]]
; SI-DAG: V_OR_B32_e32 {{v[0-9]+}}, [[HI_S_IMM]], v[[HI_VREG]]
; SI: S_ENDPGM
define void @vector_or_i64_loadimm(i64 addrspace(1)* %out, i64 addrspace(1)* %a, i64 addrspace(1)* %b) {
  %loada = load i64 addrspace(1)* %a, align 8
  %or = or i64 %loada, 22470723082367
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; FIXME: The or 0 should really be removed.
; SI-LABEL: @vector_or_i64_imm
; SI: BUFFER_LOAD_DWORDX2 v{{\[}}[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]{{\]}},
; SI: V_OR_B32_e32 {{v[0-9]+}}, 8, v[[LO_VREG]]
; SI: V_OR_B32_e32 {{v[0-9]+}}, 0, {{.*}}
; SI: S_ENDPGM
define void @vector_or_i64_imm(i64 addrspace(1)* %out, i64 addrspace(1)* %a, i64 addrspace(1)* %b) {
  %loada = load i64 addrspace(1)* %a, align 8
  %or = or i64 %loada, 8
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

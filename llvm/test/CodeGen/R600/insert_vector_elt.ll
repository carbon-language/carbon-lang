; RUN: llc -march=r600 -mcpu=SI < %s | FileCheck -check-prefix=SI %s

; FIXME: Broken on evergreen
; FIXME: For some reason the 8 and 16 vectors are being stored as
; individual elements instead of 128-bit stores.


; FIXME: Why is the constant moved into the intermediate register and
; not just directly into the vector component?

; SI-LABEL: @insertelement_v4f32_0:
; S_LOAD_DWORDX4 s{{[}}[[LOW_REG:[0-9]+]]:
; V_MOV_B32_e32
; V_MOV_B32_e32 [[CONSTREG:v[0-9]+]], 5.000000e+00
; V_MOV_B32_e32 v[[LOW_REG]], [[CONSTREG]]
; BUFFER_STORE_DWORDX4 v{{[}}[[LOW_REG]]:
define void @insertelement_v4f32_0(<4 x float> addrspace(1)* %out, <4 x float> %a) nounwind {
  %vecins = insertelement <4 x float> %a, float 5.000000e+00, i32 0
  store <4 x float> %vecins, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: @insertelement_v4f32_1:
define void @insertelement_v4f32_1(<4 x float> addrspace(1)* %out, <4 x float> %a) nounwind {
  %vecins = insertelement <4 x float> %a, float 5.000000e+00, i32 1
  store <4 x float> %vecins, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: @insertelement_v4f32_2:
define void @insertelement_v4f32_2(<4 x float> addrspace(1)* %out, <4 x float> %a) nounwind {
  %vecins = insertelement <4 x float> %a, float 5.000000e+00, i32 2
  store <4 x float> %vecins, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: @insertelement_v4f32_3:
define void @insertelement_v4f32_3(<4 x float> addrspace(1)* %out, <4 x float> %a) nounwind {
  %vecins = insertelement <4 x float> %a, float 5.000000e+00, i32 3
  store <4 x float> %vecins, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: @insertelement_v4i32_0:
define void @insertelement_v4i32_0(<4 x i32> addrspace(1)* %out, <4 x i32> %a) nounwind {
  %vecins = insertelement <4 x i32> %a, i32 999, i32 0
  store <4 x i32> %vecins, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: @dynamic_insertelement_v2f32:
; SI: V_MOV_B32_e32 [[CONST:v[0-9]+]], 5.000000e+00
; SI: V_MOVRELD_B32_e32 v[[LOW_RESULT_REG:[0-9]+]], [[CONST]]
; SI: BUFFER_STORE_DWORDX2 {{v\[}}[[LOW_RESULT_REG]]:
define void @dynamic_insertelement_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %a, i32 %b) nounwind {
  %vecins = insertelement <2 x float> %a, float 5.000000e+00, i32 %b
  store <2 x float> %vecins, <2 x float> addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: @dynamic_insertelement_v4f32:
; SI: V_MOV_B32_e32 [[CONST:v[0-9]+]], 5.000000e+00
; SI: V_MOVRELD_B32_e32 v[[LOW_RESULT_REG:[0-9]+]], [[CONST]]
; SI: BUFFER_STORE_DWORDX4 {{v\[}}[[LOW_RESULT_REG]]:
define void @dynamic_insertelement_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %a, i32 %b) nounwind {
  %vecins = insertelement <4 x float> %a, float 5.000000e+00, i32 %b
  store <4 x float> %vecins, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: @dynamic_insertelement_v8f32:
; FIXMESI: BUFFER_STORE_DWORDX4
; FIXMESI: BUFFER_STORE_DWORDX4
define void @dynamic_insertelement_v8f32(<8 x float> addrspace(1)* %out, <8 x float> %a, i32 %b) nounwind {
  %vecins = insertelement <8 x float> %a, float 5.000000e+00, i32 %b
  store <8 x float> %vecins, <8 x float> addrspace(1)* %out, align 32
  ret void
}

; SI-LABEL: @dynamic_insertelement_v16f32:
; FIXMESI: BUFFER_STORE_DWORDX4
; FIXMESI: BUFFER_STORE_DWORDX4
; FIXMESI: BUFFER_STORE_DWORDX4
; FIXMESI: BUFFER_STORE_DWORDX4
define void @dynamic_insertelement_v16f32(<16 x float> addrspace(1)* %out, <16 x float> %a, i32 %b) nounwind {
  %vecins = insertelement <16 x float> %a, float 5.000000e+00, i32 %b
  store <16 x float> %vecins, <16 x float> addrspace(1)* %out, align 64
  ret void
}

; SI-LABEL: @dynamic_insertelement_v2i32:
; SI: BUFFER_STORE_DWORDX2
define void @dynamic_insertelement_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, i32 %b) nounwind {
  %vecins = insertelement <2 x i32> %a, i32 5, i32 %b
  store <2 x i32> %vecins, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: @dynamic_insertelement_v4i32:
; SI: BUFFER_STORE_DWORDX4
define void @dynamic_insertelement_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %a, i32 %b) nounwind {
  %vecins = insertelement <4 x i32> %a, i32 5, i32 %b
  store <4 x i32> %vecins, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: @dynamic_insertelement_v8i32:
; FIXMESI: BUFFER_STORE_DWORDX4
; FIXMESI: BUFFER_STORE_DWORDX4
define void @dynamic_insertelement_v8i32(<8 x i32> addrspace(1)* %out, <8 x i32> %a, i32 %b) nounwind {
  %vecins = insertelement <8 x i32> %a, i32 5, i32 %b
  store <8 x i32> %vecins, <8 x i32> addrspace(1)* %out, align 32
  ret void
}

; SI-LABEL: @dynamic_insertelement_v16i32:
; FIXMESI: BUFFER_STORE_DWORDX4
; FIXMESI: BUFFER_STORE_DWORDX4
; FIXMESI: BUFFER_STORE_DWORDX4
; FIXMESI: BUFFER_STORE_DWORDX4
define void @dynamic_insertelement_v16i32(<16 x i32> addrspace(1)* %out, <16 x i32> %a, i32 %b) nounwind {
  %vecins = insertelement <16 x i32> %a, i32 5, i32 %b
  store <16 x i32> %vecins, <16 x i32> addrspace(1)* %out, align 64
  ret void
}


; SI-LABEL: @dynamic_insertelement_v2i16:
; FIXMESI: BUFFER_STORE_DWORDX2
define void @dynamic_insertelement_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> %a, i32 %b) nounwind {
  %vecins = insertelement <2 x i16> %a, i16 5, i32 %b
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: @dynamic_insertelement_v4i16:
; FIXMESI: BUFFER_STORE_DWORDX4
define void @dynamic_insertelement_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> %a, i32 %b) nounwind {
  %vecins = insertelement <4 x i16> %a, i16 5, i32 %b
  store <4 x i16> %vecins, <4 x i16> addrspace(1)* %out, align 16
  ret void
}


; SI-LABEL: @dynamic_insertelement_v2i8:
; FIXMESI: BUFFER_STORE_USHORT
define void @dynamic_insertelement_v2i8(<2 x i8> addrspace(1)* %out, <2 x i8> %a, i32 %b) nounwind {
  %vecins = insertelement <2 x i8> %a, i8 5, i32 %b
  store <2 x i8> %vecins, <2 x i8> addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: @dynamic_insertelement_v4i8:
; FIXMESI: BUFFER_STORE_DWORD
define void @dynamic_insertelement_v4i8(<4 x i8> addrspace(1)* %out, <4 x i8> %a, i32 %b) nounwind {
  %vecins = insertelement <4 x i8> %a, i8 5, i32 %b
  store <4 x i8> %vecins, <4 x i8> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: @dynamic_insertelement_v8i8:
; FIXMESI: BUFFER_STORE_DWORDX2
define void @dynamic_insertelement_v8i8(<8 x i8> addrspace(1)* %out, <8 x i8> %a, i32 %b) nounwind {
  %vecins = insertelement <8 x i8> %a, i8 5, i32 %b
  store <8 x i8> %vecins, <8 x i8> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: @dynamic_insertelement_v16i8:
; FIXMESI: BUFFER_STORE_DWORDX4
define void @dynamic_insertelement_v16i8(<16 x i8> addrspace(1)* %out, <16 x i8> %a, i32 %b) nounwind {
  %vecins = insertelement <16 x i8> %a, i8 5, i32 %b
  store <16 x i8> %vecins, <16 x i8> addrspace(1)* %out, align 16
  ret void
}

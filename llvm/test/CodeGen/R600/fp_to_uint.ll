; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck %s --check-prefix=EG --check-prefix=FUNC
; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck %s --check-prefix=SI --check-prefix=FUNC

; FUNC-LABEL: @fp_to_uint_v2i32
; EG: FLT_TO_UINT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; EG: FLT_TO_UINT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; SI: V_CVT_U32_F32_e32
; SI: V_CVT_U32_F32_e32

define void @fp_to_uint_v2i32(<2 x i32> addrspace(1)* %out, <2 x float> %in) {
  %result = fptoui <2 x float> %in to <2 x i32>
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @fp_to_uint_v4i32
; EG: FLT_TO_UINT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; EG: FLT_TO_UINT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: FLT_TO_UINT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; EG: FLT_TO_UINT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; SI: V_CVT_U32_F32_e32
; SI: V_CVT_U32_F32_e32
; SI: V_CVT_U32_F32_e32
; SI: V_CVT_U32_F32_e32

define void @fp_to_uint_v4i32(<4 x i32> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
  %value = load <4 x float> addrspace(1) * %in
  %result = fptoui <4 x float> %value to <4 x i32>
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}
; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck --check-prefix=SI --check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck --check-prefix=EG --check-prefix=FUNC %s

; FUNC: @fp_to_uint_i64
; EG-DAG: AND_INT
; EG-DAG: LSHR
; EG-DAG: SUB_INT
; EG-DAG: AND_INT
; EG-DAG: ASHR
; EG-DAG: AND_INT
; EG-DAG: OR_INT
; EG-DAG: SUB_INT
; EG-DAG: LSHL
; EG-DAG: LSHL
; EG-DAG: SUB_INT
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT
; EG-DAG: SETGT_INT
; EG-DAG: XOR_INT
; EG-DAG: XOR_INT
; EG: SUB_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDGE_INT
; EG-DAG: CNDGE_INT

; SI: S_ENDPGM
define void @fp_to_uint_i64(i64 addrspace(1)* %out, float %x) {
  %conv = fptoui float %x to i64
  store i64 %conv, i64 addrspace(1)* %out
  ret void
}

; FUNC: @fp_to_uint_v2i64
; EG-DAG: AND_INT
; EG-DAG: LSHR
; EG-DAG: SUB_INT
; EG-DAG: AND_INT
; EG-DAG: ASHR
; EG-DAG: AND_INT
; EG-DAG: OR_INT
; EG-DAG: SUB_INT
; EG-DAG: LSHL
; EG-DAG: LSHL
; EG-DAG: SUB_INT
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT
; EG-DAG: SETGT_INT
; EG-DAG: XOR_INT
; EG-DAG: XOR_INT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDGE_INT
; EG-DAG: CNDGE_INT
; EG-DAG: AND_INT
; EG-DAG: LSHR
; EG-DAG: SUB_INT
; EG-DAG: AND_INT
; EG-DAG: ASHR
; EG-DAG: AND_INT
; EG-DAG: OR_INT
; EG-DAG: SUB_INT
; EG-DAG: LSHL
; EG-DAG: LSHL
; EG-DAG: SUB_INT
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT
; EG-DAG: SETGT_INT
; EG-DAG: XOR_INT
; EG-DAG: XOR_INT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDGE_INT
; EG-DAG: CNDGE_INT

; SI: S_ENDPGM
define void @fp_to_uint_v2i64(<2 x i64> addrspace(1)* %out, <2 x float> %x) {
  %conv = fptoui <2 x float> %x to <2 x i64>
  store <2 x i64> %conv, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC: @fp_to_uint_v4i64
; EG-DAG: AND_INT
; EG-DAG: LSHR
; EG-DAG: SUB_INT
; EG-DAG: AND_INT
; EG-DAG: ASHR
; EG-DAG: AND_INT
; EG-DAG: OR_INT
; EG-DAG: SUB_INT
; EG-DAG: LSHL
; EG-DAG: LSHL
; EG-DAG: SUB_INT
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT
; EG-DAG: SETGT_INT
; EG-DAG: XOR_INT
; EG-DAG: XOR_INT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDGE_INT
; EG-DAG: CNDGE_INT
; EG-DAG: AND_INT
; EG-DAG: LSHR
; EG-DAG: SUB_INT
; EG-DAG: AND_INT
; EG-DAG: ASHR
; EG-DAG: AND_INT
; EG-DAG: OR_INT
; EG-DAG: SUB_INT
; EG-DAG: LSHL
; EG-DAG: LSHL
; EG-DAG: SUB_INT
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT
; EG-DAG: SETGT_INT
; EG-DAG: XOR_INT
; EG-DAG: XOR_INT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDGE_INT
; EG-DAG: CNDGE_INT
; EG-DAG: AND_INT
; EG-DAG: LSHR
; EG-DAG: SUB_INT
; EG-DAG: AND_INT
; EG-DAG: ASHR
; EG-DAG: AND_INT
; EG-DAG: OR_INT
; EG-DAG: SUB_INT
; EG-DAG: LSHL
; EG-DAG: LSHL
; EG-DAG: SUB_INT
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT
; EG-DAG: SETGT_INT
; EG-DAG: XOR_INT
; EG-DAG: XOR_INT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDGE_INT
; EG-DAG: CNDGE_INT
; EG-DAG: AND_INT
; EG-DAG: LSHR
; EG-DAG: SUB_INT
; EG-DAG: AND_INT
; EG-DAG: ASHR
; EG-DAG: AND_INT
; EG-DAG: OR_INT
; EG-DAG: SUB_INT
; EG-DAG: LSHL
; EG-DAG: LSHL
; EG-DAG: SUB_INT
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT
; EG-DAG: SETGT_INT
; EG-DAG: XOR_INT
; EG-DAG: XOR_INT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDGE_INT
; EG-DAG: CNDGE_INT

; SI: S_ENDPGM
define void @fp_to_uint_v4i64(<4 x i64> addrspace(1)* %out, <4 x float> %x) {
  %conv = fptoui <4 x float> %x to <4 x i64>
  store <4 x i64> %conv, <4 x i64> addrspace(1)* %out
  ret void
}

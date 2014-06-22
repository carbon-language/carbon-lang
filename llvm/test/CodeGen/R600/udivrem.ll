; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck --check-prefix=SI --check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck --check-prefix=EG --check-prefix=FUNC %s

; FUNC-LABEL: @test_udivrem
; EG: RECIP_UINT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG: CNDE_INT
; EG: MULHI
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG: CNDE_INT
; EG: MULHI
; EG: MULLO_INT
; EG: SUB_INT
; EG-DAG: SETGE_UINT
; EG-DAG: SETGE_UINT
; EG: AND_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT

; SI: V_RCP_IFLAG_F32_e32 [[RCP:v[0-9]+]]
; SI-DAG: V_MUL_HI_U32 [[RCP_HI:v[0-9]+]], [[RCP]]
; SI-DAG: V_MUL_LO_I32 [[RCP_LO:v[0-9]+]], [[RCP]]
; SI-DAG: V_SUB_I32_e32 [[NEG_RCP_LO:v[0-9]+]], 0, [[RCP_LO]]
; SI: V_CNDMASK_B32_e64
; SI: V_MUL_HI_U32 [[E:v[0-9]+]], {{v[0-9]+}}, [[RCP]]
; SI-DAG: V_ADD_I32_e32 [[RCP_A_E:v[0-9]+]], [[E]], [[RCP]]
; SI-DAG: V_SUBREV_I32_e32 [[RCP_S_E:v[0-9]+]], [[E]], [[RCP]]
; SI: V_CNDMASK_B32_e64
; SI: V_MUL_HI_U32 [[Quotient:v[0-9]+]]
; SI: V_MUL_LO_I32 [[Num_S_Remainder:v[0-9]+]]
; SI-DAG: V_SUB_I32_e32 [[Remainder:v[0-9]+]], {{[vs][0-9]+}}, [[Num_S_Remainder]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI: V_AND_B32_e32 [[Tmp1:v[0-9]+]]
; SI-DAG: V_ADD_I32_e32 [[Quotient_A_One:v[0-9]+]], 1, [[Quotient]]
; SI-DAG: V_SUBREV_I32_e32 [[Quotient_S_One:v[0-9]+]],
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_ADD_I32_e32 [[Remainder_A_Den:v[0-9]+]],
; SI-DAG: V_SUBREV_I32_e32 [[Remainder_S_Den:v[0-9]+]],
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI: S_ENDPGM
define void @test_udivrem(i32 addrspace(1)* %out, i32 %x, i32 %y) {
  %result0 = udiv i32 %x, %y
  store i32 %result0, i32 addrspace(1)* %out
  %result1 = urem i32 %x, %y
  store i32 %result1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @test_udivrem_v2
; EG-DAG: RECIP_UINT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: SETGE_UINT
; EG-DAG: SETGE_UINT
; EG-DAG: AND_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: RECIP_UINT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: SETGE_UINT
; EG-DAG: SETGE_UINT
; EG-DAG: AND_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT

; SI-DAG: V_RCP_IFLAG_F32_e32 [[FIRST_RCP:v[0-9]+]]
; SI-DAG: V_MUL_HI_U32 [[FIRST_RCP_HI:v[0-9]+]], [[FIRST_RCP]]
; SI-DAG: V_MUL_LO_I32 [[FIRST_RCP_LO:v[0-9]+]], [[FIRST_RCP]]
; SI-DAG: V_SUB_I32_e32 [[FIRST_NEG_RCP_LO:v[0-9]+]], 0, [[FIRST_RCP_LO]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_MUL_HI_U32 [[FIRST_E:v[0-9]+]], {{v[0-9]+}}, [[FIRST_RCP]]
; SI-DAG: V_ADD_I32_e32 [[FIRST_RCP_A_E:v[0-9]+]], [[FIRST_E]], [[FIRST_RCP]]
; SI-DAG: V_SUBREV_I32_e32 [[FIRST_RCP_S_E:v[0-9]+]], [[FIRST_E]], [[FIRST_RCP]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_MUL_HI_U32 [[FIRST_Quotient:v[0-9]+]]
; SI-DAG: V_MUL_LO_I32 [[FIRST_Num_S_Remainder:v[0-9]+]]
; SI-DAG: V_SUB_I32_e32 [[FIRST_Remainder:v[0-9]+]], {{[vs][0-9]+}}, [[FIRST_Num_S_Remainder]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_AND_B32_e32 [[FIRST_Tmp1:v[0-9]+]]
; SI-DAG: V_ADD_I32_e32 [[FIRST_Quotient_A_One:v[0-9]+]], {{.*}}, [[FIRST_Quotient]]
; SI-DAG: V_SUBREV_I32_e32 [[FIRST_Quotient_S_One:v[0-9]+]],
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_ADD_I32_e32 [[FIRST_Remainder_A_Den:v[0-9]+]],
; SI-DAG: V_SUBREV_I32_e32 [[FIRST_Remainder_S_Den:v[0-9]+]],
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_RCP_IFLAG_F32_e32 [[SECOND_RCP:v[0-9]+]]
; SI-DAG: V_MUL_HI_U32 [[SECOND_RCP_HI:v[0-9]+]], [[SECOND_RCP]]
; SI-DAG: V_MUL_LO_I32 [[SECOND_RCP_LO:v[0-9]+]], [[SECOND_RCP]]
; SI-DAG: V_SUB_I32_e32 [[SECOND_NEG_RCP_LO:v[0-9]+]], 0, [[SECOND_RCP_LO]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_MUL_HI_U32 [[SECOND_E:v[0-9]+]], {{v[0-9]+}}, [[SECOND_RCP]]
; SI-DAG: V_ADD_I32_e32 [[SECOND_RCP_A_E:v[0-9]+]], [[SECOND_E]], [[SECOND_RCP]]
; SI-DAG: V_SUBREV_I32_e32 [[SECOND_RCP_S_E:v[0-9]+]], [[SECOND_E]], [[SECOND_RCP]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_MUL_HI_U32 [[SECOND_Quotient:v[0-9]+]]
; SI-DAG: V_MUL_LO_I32 [[SECOND_Num_S_Remainder:v[0-9]+]]
; SI-DAG: V_SUB_I32_e32 [[SECOND_Remainder:v[0-9]+]], {{[vs][0-9]+}}, [[SECOND_Num_S_Remainder]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_AND_B32_e32 [[SECOND_Tmp1:v[0-9]+]]
; SI-DAG: V_ADD_I32_e32 [[SECOND_Quotient_A_One:v[0-9]+]], {{.*}}, [[SECOND_Quotient]]
; SI-DAG: V_SUBREV_I32_e32 [[SECOND_Quotient_S_One:v[0-9]+]],
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_ADD_I32_e32 [[SECOND_Remainder_A_Den:v[0-9]+]],
; SI-DAG: V_SUBREV_I32_e32 [[SECOND_Remainder_S_Den:v[0-9]+]],
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI: S_ENDPGM
define void @test_udivrem_v2(<2 x i32> addrspace(1)* %out, <2 x i32> %x, <2 x i32> %y) {
  %result0 = udiv <2 x i32> %x, %y
  store <2 x i32> %result0, <2 x i32> addrspace(1)* %out
  %result1 = urem <2 x i32> %x, %y
  store <2 x i32> %result1, <2 x i32> addrspace(1)* %out
  ret void
}


; FUNC-LABEL: @test_udivrem_v4
; EG-DAG: RECIP_UINT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: SETGE_UINT
; EG-DAG: SETGE_UINT
; EG-DAG: AND_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: RECIP_UINT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: SETGE_UINT
; EG-DAG: SETGE_UINT
; EG-DAG: AND_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: RECIP_UINT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: SETGE_UINT
; EG-DAG: SETGE_UINT
; EG-DAG: AND_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: RECIP_UINT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: SETGE_UINT
; EG-DAG: SETGE_UINT
; EG-DAG: AND_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT

; SI-DAG: V_RCP_IFLAG_F32_e32 [[FIRST_RCP:v[0-9]+]]
; SI-DAG: V_MUL_HI_U32 [[FIRST_RCP_HI:v[0-9]+]], [[FIRST_RCP]]
; SI-DAG: V_MUL_LO_I32 [[FIRST_RCP_LO:v[0-9]+]], [[FIRST_RCP]]
; SI-DAG: V_SUB_I32_e32 [[FIRST_NEG_RCP_LO:v[0-9]+]], 0, [[FIRST_RCP_LO]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_MUL_HI_U32 [[FIRST_E:v[0-9]+]], {{v[0-9]+}}, [[FIRST_RCP]]
; SI-DAG: V_ADD_I32_e32 [[FIRST_RCP_A_E:v[0-9]+]], [[FIRST_E]], [[FIRST_RCP]]
; SI-DAG: V_SUBREV_I32_e32 [[FIRST_RCP_S_E:v[0-9]+]], [[FIRST_E]], [[FIRST_RCP]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_MUL_HI_U32 [[FIRST_Quotient:v[0-9]+]]
; SI-DAG: V_MUL_LO_I32 [[FIRST_Num_S_Remainder:v[0-9]+]]
; SI-DAG: V_SUB_I32_e32 [[FIRST_Remainder:v[0-9]+]], {{[vs][0-9]+}}, [[FIRST_Num_S_Remainder]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_AND_B32_e32 [[FIRST_Tmp1:v[0-9]+]]
; SI-DAG: V_ADD_I32_e32 [[FIRST_Quotient_A_One:v[0-9]+]], {{.*}}, [[FIRST_Quotient]]
; SI-DAG: V_SUBREV_I32_e32 [[FIRST_Quotient_S_One:v[0-9]+]],
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_ADD_I32_e32 [[FIRST_Remainder_A_Den:v[0-9]+]],
; SI-DAG: V_SUBREV_I32_e32 [[FIRST_Remainder_S_Den:v[0-9]+]],
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_RCP_IFLAG_F32_e32 [[SECOND_RCP:v[0-9]+]]
; SI-DAG: V_MUL_HI_U32 [[SECOND_RCP_HI:v[0-9]+]], [[SECOND_RCP]]
; SI-DAG: V_MUL_LO_I32 [[SECOND_RCP_LO:v[0-9]+]], [[SECOND_RCP]]
; SI-DAG: V_SUB_I32_e32 [[SECOND_NEG_RCP_LO:v[0-9]+]], 0, [[SECOND_RCP_LO]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_MUL_HI_U32 [[SECOND_E:v[0-9]+]], {{v[0-9]+}}, [[SECOND_RCP]]
; SI-DAG: V_ADD_I32_e32 [[SECOND_RCP_A_E:v[0-9]+]], [[SECOND_E]], [[SECOND_RCP]]
; SI-DAG: V_SUBREV_I32_e32 [[SECOND_RCP_S_E:v[0-9]+]], [[SECOND_E]], [[SECOND_RCP]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_MUL_HI_U32 [[SECOND_Quotient:v[0-9]+]]
; SI-DAG: V_MUL_LO_I32 [[SECOND_Num_S_Remainder:v[0-9]+]]
; SI-DAG: V_SUB_I32_e32 [[SECOND_Remainder:v[0-9]+]], {{[vs][0-9]+}}, [[SECOND_Num_S_Remainder]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_AND_B32_e32 [[SECOND_Tmp1:v[0-9]+]]
; SI-DAG: V_ADD_I32_e32 [[SECOND_Quotient_A_One:v[0-9]+]], {{.*}}, [[SECOND_Quotient]]
; SI-DAG: V_SUBREV_I32_e32 [[SECOND_Quotient_S_One:v[0-9]+]],
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_ADD_I32_e32 [[SECOND_Remainder_A_Den:v[0-9]+]],
; SI-DAG: V_SUBREV_I32_e32 [[SECOND_Remainder_S_Den:v[0-9]+]],
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_RCP_IFLAG_F32_e32 [[THIRD_RCP:v[0-9]+]]
; SI-DAG: V_MUL_HI_U32 [[THIRD_RCP_HI:v[0-9]+]], [[THIRD_RCP]]
; SI-DAG: V_MUL_LO_I32 [[THIRD_RCP_LO:v[0-9]+]], [[THIRD_RCP]]
; SI-DAG: V_SUB_I32_e32 [[THIRD_NEG_RCP_LO:v[0-9]+]], 0, [[THIRD_RCP_LO]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_MUL_HI_U32 [[THIRD_E:v[0-9]+]], {{v[0-9]+}}, [[THIRD_RCP]]
; SI-DAG: V_ADD_I32_e32 [[THIRD_RCP_A_E:v[0-9]+]], [[THIRD_E]], [[THIRD_RCP]]
; SI-DAG: V_SUBREV_I32_e32 [[THIRD_RCP_S_E:v[0-9]+]], [[THIRD_E]], [[THIRD_RCP]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_MUL_HI_U32 [[THIRD_Quotient:v[0-9]+]]
; SI-DAG: V_MUL_LO_I32 [[THIRD_Num_S_Remainder:v[0-9]+]]
; SI-DAG: V_SUB_I32_e32 [[THIRD_Remainder:v[0-9]+]], {{[vs][0-9]+}}, [[THIRD_Num_S_Remainder]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_AND_B32_e32 [[THIRD_Tmp1:v[0-9]+]]
; SI-DAG: V_ADD_I32_e32 [[THIRD_Quotient_A_One:v[0-9]+]], {{.*}}, [[THIRD_Quotient]]
; SI-DAG: V_SUBREV_I32_e32 [[THIRD_Quotient_S_One:v[0-9]+]],
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_ADD_I32_e32 [[THIRD_Remainder_A_Den:v[0-9]+]],
; SI-DAG: V_SUBREV_I32_e32 [[THIRD_Remainder_S_Den:v[0-9]+]],
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_RCP_IFLAG_F32_e32 [[FOURTH_RCP:v[0-9]+]]
; SI-DAG: V_MUL_HI_U32 [[FOURTH_RCP_HI:v[0-9]+]], [[FOURTH_RCP]]
; SI-DAG: V_MUL_LO_I32 [[FOURTH_RCP_LO:v[0-9]+]], [[FOURTH_RCP]]
; SI-DAG: V_SUB_I32_e32 [[FOURTH_NEG_RCP_LO:v[0-9]+]], 0, [[FOURTH_RCP_LO]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_MUL_HI_U32 [[FOURTH_E:v[0-9]+]], {{v[0-9]+}}, [[FOURTH_RCP]]
; SI-DAG: V_ADD_I32_e32 [[FOURTH_RCP_A_E:v[0-9]+]], [[FOURTH_E]], [[FOURTH_RCP]]
; SI-DAG: V_SUBREV_I32_e32 [[FOURTH_RCP_S_E:v[0-9]+]], [[FOURTH_E]], [[FOURTH_RCP]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_MUL_HI_U32 [[FOURTH_Quotient:v[0-9]+]]
; SI-DAG: V_MUL_LO_I32 [[FOURTH_Num_S_Remainder:v[0-9]+]]
; SI-DAG: V_SUB_I32_e32 [[FOURTH_Remainder:v[0-9]+]], {{[vs][0-9]+}}, [[FOURTH_Num_S_Remainder]]
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_AND_B32_e32 [[FOURTH_Tmp1:v[0-9]+]]
; SI-DAG: V_ADD_I32_e32 [[FOURTH_Quotient_A_One:v[0-9]+]], {{.*}}, [[FOURTH_Quotient]]
; SI-DAG: V_SUBREV_I32_e32 [[FOURTH_Quotient_S_One:v[0-9]+]],
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_ADD_I32_e32 [[FOURTH_Remainder_A_Den:v[0-9]+]],
; SI-DAG: V_SUBREV_I32_e32 [[FOURTH_Remainder_S_Den:v[0-9]+]],
; SI-DAG: V_CNDMASK_B32_e64
; SI-DAG: V_CNDMASK_B32_e64
; SI: S_ENDPGM
define void @test_udivrem_v4(<4 x i32> addrspace(1)* %out, <4 x i32> %x, <4 x i32> %y) {
  %result0 = udiv <4 x i32> %x, %y
  store <4 x i32> %result0, <4 x i32> addrspace(1)* %out
  %result1 = urem <4 x i32> %x, %y
  store <4 x i32> %result1, <4 x i32> addrspace(1)* %out
  ret void
}

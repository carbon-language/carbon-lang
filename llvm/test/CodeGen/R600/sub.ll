;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG --check-prefix=FUNC %s
;RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI --check-prefix=FUNC %s

;FUNC-LABEL: @test2
;EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI: V_SUB_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: V_SUB_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @test2(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32> addrspace(1) * %in
  %b = load <2 x i32> addrspace(1) * %b_ptr
  %result = sub <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

;FUNC-LABEL: @test4
;EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI: V_SUB_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: V_SUB_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: V_SUB_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
;SI: V_SUB_I32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @test4(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32> addrspace(1) * %in
  %b = load <4 x i32> addrspace(1) * %b_ptr
  %result = sub <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

;FUNC_LABEL: @test5

;EG-DAG: SETGE_UINT
;EG-DAG: CNDE_INT
;EG-DAG: SUB_INT
;EG-DAG: SUB_INT
;EG-DAG: SUB_INT

;SI: S_XOR_B64
;SI-DAG: S_ADD_I32
;SI-DAG: S_ADDC_U32
;SI-DAG: S_ADD_I32
;SI-DAG: S_ADDC_U32

define void @test5(i64 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %0 = sub i64 %a, %b
  store i64 %0, i64 addrspace(1)* %out
  ret void
}

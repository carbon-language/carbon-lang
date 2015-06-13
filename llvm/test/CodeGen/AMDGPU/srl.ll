; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}lshr_i32:
; SI: v_lshrrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_lshrrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; EG: LSHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
define void @lshr_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %b_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %a = load i32, i32 addrspace(1)* %in
  %b = load i32, i32 addrspace(1)* %b_ptr
  %result = lshr i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}lshr_v2i32:
; SI: v_lshr_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_lshr_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

; VI: v_lshrrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_lshrrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

; EG: LSHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: LSHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
define void @lshr_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32>, <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32>, <2 x i32> addrspace(1)* %in
  %b = load <2 x i32>, <2 x i32> addrspace(1)* %b_ptr
  %result = lshr <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}lshr_v4i32:
; SI: v_lshr_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_lshr_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_lshr_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_lshr_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

; VI: v_lshrrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_lshrrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_lshrrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_lshrrev_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

; EG: LSHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: LSHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: LSHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: LSHR {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
define void @lshr_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32>, <4 x i32> addrspace(1)* %in
  %b = load <4 x i32>, <4 x i32> addrspace(1)* %b_ptr
  %result = lshr <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}lshr_i64:
; SI: v_lshr_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
; VI: v_lshrrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

; EG: SUB_INT {{\*? *}}[[COMPSH:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHIFT:T[0-9]+\.[XYZW]]]
; EG: LSHL {{\* *}}[[TEMP:T[0-9]+\.[XYZW]]], [[OPHI:T[0-9]+\.[XYZW]]], {{[[COMPSH]]|PV.[XYZW]}}
; EG: LSHL {{\*? *}}[[OVERF:T[0-9]+\.[XYZW]]], {{[[TEMP]]|PV.[XYZW]}}, 1
; EG-DAG: ADD_INT {{\*? *}}[[BIGSH:T[0-9]+\.[XYZW]]], [[SHIFT]], literal
; EG-DAG: LSHR {{\*? *}}[[LOSMTMP:T[0-9]+\.[XYZW]]], [[OPLO:T[0-9]+\.[XYZW]]], [[SHIFT]]
; EG-DAG: OR_INT {{\*? *}}[[LOSM:T[0-9]+\.[XYZW]]], {{[[LOSMTMP]]|PV.[XYZW]}}, {{[[OVERF]]|PV.[XYZW]}}
; EG-DAG: LSHR {{\*? *}}[[HISM:T[0-9]+\.[XYZW]]], [[OPHI]], {{PS|[[SHIFT]]}}
; EG-DAG: LSHR {{\*? *}}[[LOBIG:T[0-9]+\.[XYZW]]], [[OPHI]], {{PS|[[SHIFT]]}}
; EG-DAG: SETGT_UINT {{\*? *}}[[RESC:T[0-9]+\.[XYZW]]], [[SHIFT]], literal
; EG-DAG: CNDE_INT {{\*? *}}[[RESLO:T[0-9]+\.[XYZW]]], {{T[0-9]+\.[XYZW]}}
; EG-DAG: CNDE_INT {{\*? *}}[[RESHI:T[0-9]+\.[XYZW]]], {{T[0-9]+\.[XYZW], .*}}, 0.0
define void @lshr_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %b_ptr = getelementptr i64, i64 addrspace(1)* %in, i64 1
  %a = load i64, i64 addrspace(1)* %in
  %b = load i64, i64 addrspace(1)* %b_ptr
  %result = lshr i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}lshr_v2i64:
; SI: v_lshr_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
; SI: v_lshr_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}

; VI: v_lshrrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
; VI: v_lshrrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

; EG-DAG: SUB_INT {{\*? *}}[[COMPSHA:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHA:T[0-9]+\.[XYZW]]]
; EG-DAG: SUB_INT {{\*? *}}[[COMPSHB:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHB:T[0-9]+\.[XYZW]]]
; EG-DAG: LSHL {{\*? *}}[[COMPSHA]]
; EG-DAG: LSHL {{\*? *}}[[COMPSHB]]
; EG-DAG: LSHL {{.*}}, 1
; EG-DAG: LSHL {{.*}}, 1
; EG-DAG: LSHR {{.*}}, [[SHA]]
; EG-DAG: LSHR {{.*}}, [[SHB]]
; EG-DAG: LSHR {{.*}}, [[SHA]]
; EG-DAG: LSHR {{.*}}, [[SHB]]
; EG-DAG: OR_INT
; EG-DAG: OR_INT
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHA:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHB:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHA]], literal
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHB]], literal
; EG-DAG: CNDE_INT {{.*}}, 0.0
; EG-DAG: CNDE_INT {{.*}}, 0.0
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
define void @lshr_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i64>, <2 x i64> addrspace(1)* %in, i64 1
  %a = load <2 x i64>, <2 x i64> addrspace(1)* %in
  %b = load <2 x i64>, <2 x i64> addrspace(1)* %b_ptr
  %result = lshr <2 x i64> %a, %b
  store <2 x i64> %result, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}lshr_v4i64:
; SI: v_lshr_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
; SI: v_lshr_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
; SI: v_lshr_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}
; SI: v_lshr_b64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v[0-9]+}}

; VI: v_lshrrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
; VI: v_lshrrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
; VI: v_lshrrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}
; VI: v_lshrrev_b64 {{v\[[0-9]+:[0-9]+\], v[0-9]+, v\[[0-9]+:[0-9]+\]}}

; EG-DAG: SUB_INT {{\*? *}}[[COMPSHA:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHA:T[0-9]+\.[XYZW]]]
; EG-DAG: SUB_INT {{\*? *}}[[COMPSHB:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHB:T[0-9]+\.[XYZW]]]
; EG-DAG: SUB_INT {{\*? *}}[[COMPSHC:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHC:T[0-9]+\.[XYZW]]]
; EG-DAG: SUB_INT {{\*? *}}[[COMPSHD:T[0-9]+\.[XYZW]]], {{literal.[xy]}}, [[SHD:T[0-9]+\.[XYZW]]]
; EG-DAG: LSHL {{\*? *}}[[COMPSHA]]
; EG-DAG: LSHL {{\*? *}}[[COMPSHB]]
; EG-DAG: LSHL {{\*? *}}[[COMPSHC]]
; EG-DAG: LSHL {{\*? *}}[[COMPSHD]]
; EG-DAG: LSHL {{.*}}, 1
; EG-DAG: LSHL {{.*}}, 1
; EG-DAG: LSHL {{.*}}, 1
; EG-DAG: LSHL {{.*}}, 1
; EG-DAG: LSHR {{.*}}, [[SHA]]
; EG-DAG: LSHR {{.*}}, [[SHB]]
; EG-DAG: LSHR {{.*}}, [[SHC]]
; EG-DAG: LSHR {{.*}}, [[SHD]]
; EG-DAG: LSHR {{.*}}, [[SHA]]
; EG-DAG: LSHR {{.*}}, [[SHB]]
; EG-DAG: LSHR {{.*}}, [[SHC]]
; EG-DAG: LSHR {{.*}}, [[SHD]]
; EG-DAG: OR_INT
; EG-DAG: OR_INT
; EG-DAG: OR_INT
; EG-DAG: OR_INT
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHA:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHB:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHC:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: ADD_INT  {{\*? *}}[[BIGSHD:T[0-9]+\.[XYZW]]]{{.*}}, literal
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHA]], literal
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHB]], literal
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHC]], literal
; EG-DAG: SETGT_UINT {{\*? *T[0-9]\.[XYZW]}}, [[SHD]], literal
; EG-DAG: CNDE_INT {{.*}}, 0.0
; EG-DAG: CNDE_INT {{.*}}, 0.0
; EG-DAG: CNDE_INT {{.*}}, 0.0
; EG-DAG: CNDE_INT {{.*}}, 0.0
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
define void @lshr_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i64>, <4 x i64> addrspace(1)* %in, i64 1
  %a = load <4 x i64>, <4 x i64> addrspace(1)* %in
  %b = load <4 x i64>, <4 x i64> addrspace(1)* %b_ptr
  %result = lshr <4 x i64> %a, %b
  store <4 x i64> %result, <4 x i64> addrspace(1)* %out
  ret void
}

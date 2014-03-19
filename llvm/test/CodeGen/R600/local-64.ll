; RUN: llc -march=r600 -mcpu=SI < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: @local_i32_load
; SI: DS_READ_B32 [[REG:v[0-9]+]], v{{[0-9]+}}, 28, [M0]
; SI: BUFFER_STORE_DWORD [[REG]],
define void @local_i32_load(i32 addrspace(1)* %out, i32 addrspace(3)* %in) nounwind {
  %gep = getelementptr i32 addrspace(3)* %in, i32 7
  %val = load i32 addrspace(3)* %gep, align 4
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: @local_i32_load_0_offset
; SI: DS_READ_B32 [[REG:v[0-9]+]], v{{[0-9]+}}, 0, [M0]
; SI: BUFFER_STORE_DWORD [[REG]],
define void @local_i32_load_0_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %in) nounwind {
  %val = load i32 addrspace(3)* %in, align 4
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: @local_i8_load_i16_max_offset
; SI-NOT: ADD
; SI: DS_READ_U8 [[REG:v[0-9]+]], {{v[0-9]+}}, -1, [M0]
; SI: BUFFER_STORE_BYTE [[REG]],
define void @local_i8_load_i16_max_offset(i8 addrspace(1)* %out, i8 addrspace(3)* %in) nounwind {
  %gep = getelementptr i8 addrspace(3)* %in, i32 65535
  %val = load i8 addrspace(3)* %gep, align 4
  store i8 %val, i8 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: @local_i8_load_over_i16_max_offset
; SI: S_ADD_I32 [[ADDR:s[0-9]+]], s{{[0-9]+}}, 65536
; SI: V_MOV_B32_e32 [[VREGADDR:v[0-9]+]], [[ADDR]]
; SI: DS_READ_U8 [[REG:v[0-9]+]], [[VREGADDR]], 0, [M0]
; SI: BUFFER_STORE_BYTE [[REG]],
define void @local_i8_load_over_i16_max_offset(i8 addrspace(1)* %out, i8 addrspace(3)* %in) nounwind {
  %gep = getelementptr i8 addrspace(3)* %in, i32 65536
  %val = load i8 addrspace(3)* %gep, align 4
  store i8 %val, i8 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: @local_i64_load
; SI-NOT: ADD
; SI: DS_READ_B64 [[REG:v[[0-9]+:[0-9]+]]], v{{[0-9]+}}, 56, [M0]
; SI: BUFFER_STORE_DWORDX2 [[REG]],
define void @local_i64_load(i64 addrspace(1)* %out, i64 addrspace(3)* %in) nounwind {
  %gep = getelementptr i64 addrspace(3)* %in, i32 7
  %val = load i64 addrspace(3)* %gep, align 8
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: @local_i64_load_0_offset
; SI: DS_READ_B64 [[REG:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}, 0, [M0]
; SI: BUFFER_STORE_DWORDX2 [[REG]],
define void @local_i64_load_0_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %in) nounwind {
  %val = load i64 addrspace(3)* %in, align 8
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: @local_f64_load
; SI-NOT: ADD
; SI: DS_READ_B64 [[REG:v[[0-9]+:[0-9]+]]], v{{[0-9]+}}, 56, [M0]
; SI: BUFFER_STORE_DWORDX2 [[REG]],
define void @local_f64_load(double addrspace(1)* %out, double addrspace(3)* %in) nounwind {
  %gep = getelementptr double addrspace(3)* %in, i32 7
  %val = load double addrspace(3)* %gep, align 8
  store double %val, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: @local_f64_load_0_offset
; SI: DS_READ_B64 [[REG:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}, 0, [M0]
; SI: BUFFER_STORE_DWORDX2 [[REG]],
define void @local_f64_load_0_offset(double addrspace(1)* %out, double addrspace(3)* %in) nounwind {
  %val = load double addrspace(3)* %in, align 8
  store double %val, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: @local_i64_store
; SI-NOT: ADD
; SI: DS_WRITE_B64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, 56 [M0]
define void @local_i64_store(i64 addrspace(3)* %out) nounwind {
  %gep = getelementptr i64 addrspace(3)* %out, i32 7
  store i64 5678, i64 addrspace(3)* %gep, align 8
  ret void
}

; SI-LABEL: @local_i64_store_0_offset
; SI-NOT: ADD
; SI: DS_WRITE_B64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, 0 [M0]
define void @local_i64_store_0_offset(i64 addrspace(3)* %out) nounwind {
  store i64 1234, i64 addrspace(3)* %out, align 8
  ret void
}

; SI-LABEL: @local_f64_store
; SI-NOT: ADD
; SI: DS_WRITE_B64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, 56 [M0]
define void @local_f64_store(double addrspace(3)* %out) nounwind {
  %gep = getelementptr double addrspace(3)* %out, i32 7
  store double 16.0, double addrspace(3)* %gep, align 8
  ret void
}

; SI-LABEL: @local_f64_store_0_offset
; SI: DS_WRITE_B64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, 0 [M0]
define void @local_f64_store_0_offset(double addrspace(3)* %out) nounwind {
  store double 20.0, double addrspace(3)* %out, align 8
  ret void
}

; SI-LABEL: @local_v2i64_store
; SI-NOT: ADD
; SI-DAG: DS_WRITE_B64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, 120 [M0]
; SI-DAG: DS_WRITE_B64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, 112 [M0]
define void @local_v2i64_store(<2 x i64> addrspace(3)* %out) nounwind {
  %gep = getelementptr <2 x i64> addrspace(3)* %out, i32 7
  store <2 x i64> <i64 5678, i64 5678>, <2 x i64> addrspace(3)* %gep, align 16
  ret void
}

; SI-LABEL: @local_v2i64_store_0_offset
; SI-NOT: ADD
; SI-DAG: DS_WRITE_B64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, 8 [M0]
; SI-DAG: DS_WRITE_B64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, 0 [M0]
define void @local_v2i64_store_0_offset(<2 x i64> addrspace(3)* %out) nounwind {
  store <2 x i64> <i64 1234, i64 1234>, <2 x i64> addrspace(3)* %out, align 16
  ret void
}

; SI-LABEL: @local_v4i64_store
; SI-NOT: ADD
; SI-DAG: DS_WRITE_B64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, 248 [M0]
; SI-DAG: DS_WRITE_B64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, 240 [M0]
; SI-DAG: DS_WRITE_B64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, 232 [M0]
; SI-DAG: DS_WRITE_B64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, 224 [M0]
define void @local_v4i64_store(<4 x i64> addrspace(3)* %out) nounwind {
  %gep = getelementptr <4 x i64> addrspace(3)* %out, i32 7
  store <4 x i64> <i64 5678, i64 5678, i64 5678, i64 5678>, <4 x i64> addrspace(3)* %gep, align 16
  ret void
}

; SI-LABEL: @local_v4i64_store_0_offset
; SI-NOT: ADD
; SI-DAG: DS_WRITE_B64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, 24 [M0]
; SI-DAG: DS_WRITE_B64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, 16 [M0]
; SI-DAG: DS_WRITE_B64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, 8 [M0]
; SI-DAG: DS_WRITE_B64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, 0 [M0]
define void @local_v4i64_store_0_offset(<4 x i64> addrspace(3)* %out) nounwind {
  store <4 x i64> <i64 1234, i64 1234, i64 1234, i64 1234>, <4 x i64> addrspace(3)* %out, align 16
  ret void
}

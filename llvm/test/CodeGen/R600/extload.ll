; RUN: llc -march=r600 -mcpu=cypress < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=SI < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: @anyext_load_i8:
; EG: AND_INT
; EG: 255
define void @anyext_load_i8(i8 addrspace(1)* nocapture noalias %out, i8 addrspace(1)* nocapture noalias %src) nounwind {
  %cast = bitcast i8 addrspace(1)* %src to i32 addrspace(1)*
  %load = load i32 addrspace(1)* %cast, align 1
  %x = bitcast i32 %load to <4 x i8>
  %castOut = bitcast i8 addrspace(1)* %out to <4 x i8> addrspace(1)*
  store <4 x i8> %x, <4 x i8> addrspace(1)* %castOut, align 1
  ret void
}

; FUNC-LABEL: @anyext_load_i16:
; EG: AND_INT
; EG: AND_INT
; EG-DAG: 65535
; EG-DAG: -65536
define void @anyext_load_i16(i16 addrspace(1)* nocapture noalias %out, i16 addrspace(1)* nocapture noalias %src) nounwind {
  %cast = bitcast i16 addrspace(1)* %src to i32 addrspace(1)*
  %load = load i32 addrspace(1)* %cast, align 1
  %x = bitcast i32 %load to <2 x i16>
  %castOut = bitcast i16 addrspace(1)* %out to <2 x i16> addrspace(1)*
  store <2 x i16> %x, <2 x i16> addrspace(1)* %castOut, align 1
  ret void
}

; FUNC-LABEL: @anyext_load_lds_i8:
; EG: AND_INT
; EG: 255
define void @anyext_load_lds_i8(i8 addrspace(3)* nocapture noalias %out, i8 addrspace(3)* nocapture noalias %src) nounwind {
  %cast = bitcast i8 addrspace(3)* %src to i32 addrspace(3)*
  %load = load i32 addrspace(3)* %cast, align 1
  %x = bitcast i32 %load to <4 x i8>
  %castOut = bitcast i8 addrspace(3)* %out to <4 x i8> addrspace(3)*
  store <4 x i8> %x, <4 x i8> addrspace(3)* %castOut, align 1
  ret void
}

; FUNC-LABEL: @anyext_load_lds_i16:
; EG: AND_INT
; EG: AND_INT
; EG-DAG: 65535
; EG-DAG: -65536
define void @anyext_load_lds_i16(i16 addrspace(3)* nocapture noalias %out, i16 addrspace(3)* nocapture noalias %src) nounwind {
  %cast = bitcast i16 addrspace(3)* %src to i32 addrspace(3)*
  %load = load i32 addrspace(3)* %cast, align 1
  %x = bitcast i32 %load to <2 x i16>
  %castOut = bitcast i16 addrspace(3)* %out to <2 x i16> addrspace(3)*
  store <2 x i16> %x, <2 x i16> addrspace(3)* %castOut, align 1
  ret void
}

; FUNC-LABEL: @sextload_global_i8_to_i64
; SI: BUFFER_LOAD_SBYTE [[LOAD:v[0-9]+]],
; SI: V_ASHRREV_I32_e32 v{{[0-9]+}}, 31, [[LOAD]]
; SI: BUFFER_STORE_DWORDX2
define void @sextload_global_i8_to_i64(i64 addrspace(1)* %out, i8 addrspace(1)* %in) nounwind {
  %a = load i8 addrspace(1)* %in, align 8
  %ext = sext i8 %a to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @sextload_global_i16_to_i64
; SI: BUFFER_LOAD_SSHORT [[LOAD:v[0-9]+]],
; SI: V_ASHRREV_I32_e32 v{{[0-9]+}}, 31, [[LOAD]]
; SI: BUFFER_STORE_DWORDX2
define void @sextload_global_i16_to_i64(i64 addrspace(1)* %out, i16 addrspace(1)* %in) nounwind {
  %a = load i16 addrspace(1)* %in, align 8
  %ext = sext i16 %a to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @sextload_global_i32_to_i64
; SI: BUFFER_LOAD_DWORD [[LOAD:v[0-9]+]],
; SI: V_ASHRREV_I32_e32 v{{[0-9]+}}, 31, [[LOAD]]
; SI: BUFFER_STORE_DWORDX2
define void @sextload_global_i32_to_i64(i64 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %a = load i32 addrspace(1)* %in, align 8
  %ext = sext i32 %a to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @zextload_global_i8_to_i64
; SI: S_MOV_B32 [[ZERO:s[0-9]+]], 0
; SI: BUFFER_LOAD_UBYTE [[LOAD:v[0-9]+]],
; SI: V_MOV_B32_e32 {{v[0-9]+}}, [[ZERO]]
; SI: BUFFER_STORE_DWORDX2
define void @zextload_global_i8_to_i64(i64 addrspace(1)* %out, i8 addrspace(1)* %in) nounwind {
  %a = load i8 addrspace(1)* %in, align 8
  %ext = zext i8 %a to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @zextload_global_i16_to_i64
; SI: S_MOV_B32 [[ZERO:s[0-9]+]], 0
; SI: BUFFER_LOAD_USHORT [[LOAD:v[0-9]+]],
; SI: V_MOV_B32_e32 {{v[0-9]+}}, [[ZERO]]
; SI: BUFFER_STORE_DWORDX2
define void @zextload_global_i16_to_i64(i64 addrspace(1)* %out, i16 addrspace(1)* %in) nounwind {
  %a = load i16 addrspace(1)* %in, align 8
  %ext = zext i16 %a to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @zextload_global_i32_to_i64
; SI: S_MOV_B32 [[ZERO:s[0-9]+]], 0
; SI: BUFFER_LOAD_DWORD [[LOAD:v[0-9]+]],
; SI: V_MOV_B32_e32 {{v[0-9]+}}, [[ZERO]]
; SI: BUFFER_STORE_DWORDX2
define void @zextload_global_i32_to_i64(i64 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %a = load i32 addrspace(1)* %in, align 8
  %ext = zext i32 %a to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

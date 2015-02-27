; RUN: llc -march=r600 -mcpu=cypress < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs< %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}anyext_load_i8:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+.[XYZW]]],
; EG: VTX_READ_32 [[VAL]]

define void @anyext_load_i8(i8 addrspace(1)* nocapture noalias %out, i8 addrspace(1)* nocapture noalias %src) nounwind {
  %cast = bitcast i8 addrspace(1)* %src to i32 addrspace(1)*
  %load = load i32, i32 addrspace(1)* %cast, align 1
  %x = bitcast i32 %load to <4 x i8>
  %castOut = bitcast i8 addrspace(1)* %out to <4 x i8> addrspace(1)*
  store <4 x i8> %x, <4 x i8> addrspace(1)* %castOut, align 1
  ret void
}

; FUNC-LABEL: {{^}}anyext_load_i16:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+.[XYZW]]],
; EG: VTX_READ_32 [[VAL]]

define void @anyext_load_i16(i16 addrspace(1)* nocapture noalias %out, i16 addrspace(1)* nocapture noalias %src) nounwind {
  %cast = bitcast i16 addrspace(1)* %src to i32 addrspace(1)*
  %load = load i32, i32 addrspace(1)* %cast, align 1
  %x = bitcast i32 %load to <2 x i16>
  %castOut = bitcast i16 addrspace(1)* %out to <2 x i16> addrspace(1)*
  store <2 x i16> %x, <2 x i16> addrspace(1)* %castOut, align 1
  ret void
}

; FUNC-LABEL: {{^}}anyext_load_lds_i8:
; EG: LDS_READ_RET {{.*}}, [[VAL:T[0-9]+.[XYZW]]]
; EG: LDS_WRITE * [[VAL]]
define void @anyext_load_lds_i8(i8 addrspace(3)* nocapture noalias %out, i8 addrspace(3)* nocapture noalias %src) nounwind {
  %cast = bitcast i8 addrspace(3)* %src to i32 addrspace(3)*
  %load = load i32, i32 addrspace(3)* %cast, align 1
  %x = bitcast i32 %load to <4 x i8>
  %castOut = bitcast i8 addrspace(3)* %out to <4 x i8> addrspace(3)*
  store <4 x i8> %x, <4 x i8> addrspace(3)* %castOut, align 1
  ret void
}

; FUNC-LABEL: {{^}}anyext_load_lds_i16:
; EG: LDS_READ_RET {{.*}}, [[VAL:T[0-9]+.[XYZW]]]
; EG: LDS_WRITE * [[VAL]]
define void @anyext_load_lds_i16(i16 addrspace(3)* nocapture noalias %out, i16 addrspace(3)* nocapture noalias %src) nounwind {
  %cast = bitcast i16 addrspace(3)* %src to i32 addrspace(3)*
  %load = load i32, i32 addrspace(3)* %cast, align 1
  %x = bitcast i32 %load to <2 x i16>
  %castOut = bitcast i16 addrspace(3)* %out to <2 x i16> addrspace(3)*
  store <2 x i16> %x, <2 x i16> addrspace(3)* %castOut, align 1
  ret void
}

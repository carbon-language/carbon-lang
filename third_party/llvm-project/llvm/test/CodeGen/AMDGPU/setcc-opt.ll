; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}sext_bool_icmp_eq_0:
; GCN-NOT: v_cmp
; GCN: s_cmp_lg_u32
; GCN: s_cselect_b64 [[CC:[^,]+]], -1, 0
; GCN: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, [[CC]]
; GCN-NEXT:buffer_store_byte [[RESULT]]
; GCN-NEXT: s_endpgm

; EG: SETNE_INT * [[CMP:T[0-9]+]].[[CMPCHAN:[XYZW]]], KC0[2].Z, KC0[2].W
; EG: AND_INT T{{[0-9]+.[XYZW]}}, PS, 1
define amdgpu_kernel void @sext_bool_icmp_eq_0(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp eq i32 %a, %b
  %ext = sext i1 %icmp0 to i32
  %icmp1 = icmp eq i32 %ext, 0
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sext_bool_icmp_ne_0:
; GCN-NOT: v_cmp
; GCN: s_cmp_lg_u32
; GCN: s_cselect_b64 [[CC:[^,]+]], -1, 0
; GCN: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, [[CC]]
; GCN-NEXT: buffer_store_byte [[RESULT]]
; GCN-NEXT: s_endpgm

; EG: SETNE_INT * [[CMP:T[0-9]+]].[[CMPCHAN:[XYZW]]], KC0[2].Z, KC0[2].W
; EG: AND_INT T{{[0-9]+.[XYZW]}}, PS, 1
define amdgpu_kernel void @sext_bool_icmp_ne_0(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = sext i1 %icmp0 to i32
  %icmp1 = icmp ne i32 %ext, 0
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sext_bool_icmp_eq_neg1:
; GCN-NOT: v_cmp
; GCN: s_cmp_eq_u32
; GCN: s_cselect_b64 [[CC:[^,]+]], -1, 0
; GCN: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, [[CC]]
; GCN-NEXT: buffer_store_byte [[RESULT]]
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @sext_bool_icmp_eq_neg1(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp eq i32 %a, %b
  %ext = sext i1 %icmp0 to i32
  %icmp1 = icmp eq i32 %ext, -1
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sext_bool_icmp_ne_neg1:
; GCN-NOT: v_cmp
; GCN: s_cmp_eq_u32
; GCN: s_cselect_b64 [[CC:[^,]+]], -1, 0
; GCN: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, [[CC]]
; GCN-NEXT: buffer_store_byte [[RESULT]]
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @sext_bool_icmp_ne_neg1(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = sext i1 %icmp0 to i32
  %icmp1 = icmp ne i32 %ext, -1
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zext_bool_icmp_eq_0:
; GCN-NOT: v_cmp
; GCN: s_cmp_lg_u32
; GCN: s_cselect_b64 [[CC:[^,]+]], -1, 0
; GCN: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, [[CC]]
; GCN-NEXT: buffer_store_byte [[RESULT]]
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @zext_bool_icmp_eq_0(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp eq i32 %a, %b
  %ext = zext i1 %icmp0 to i32
  %icmp1 = icmp eq i32 %ext, 0
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zext_bool_icmp_ne_0:
; GCN-NOT: v_cmp
; GCN: s_cmp_lg_u32
; GCN: s_cselect_b64 [[CC:[^,]+]], -1, 0
; GCN: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, [[CC]]
; GCN-NEXT: buffer_store_byte [[RESULT]]
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @zext_bool_icmp_ne_0(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = zext i1 %icmp0 to i32
  %icmp1 = icmp ne i32 %ext, 0
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zext_bool_icmp_eq_1:
; GCN-NOT: v_cmp
; GCN: s_cmp_eq_u32
; GCN: s_cselect_b64 [[CC:[^,]+]], -1, 0
; GCN: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, [[CC]]
; GCN-NEXT: buffer_store_byte [[RESULT]]
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @zext_bool_icmp_eq_1(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp eq i32 %a, %b
  %ext = zext i1 %icmp0 to i32
  %icmp1 = icmp eq i32 %ext, 1
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zext_bool_icmp_ne_1:
; GCN-NOT: v_cmp
; GCN: s_cmp_eq_u32
; GCN: s_cselect_b64 [[CC:[^,]+]], -1, 0
; GCN: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, [[CC]]
; GCN-NEXT: buffer_store_byte [[RESULT]]
define amdgpu_kernel void @zext_bool_icmp_ne_1(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = zext i1 %icmp0 to i32
  %icmp1 = icmp ne i32 %ext, 1
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; Reduces to false:
; FUNC-LABEL: {{^}}zext_bool_icmp_eq_neg1:
; GCN: v_mov_b32_e32 [[TMP:v[0-9]+]], 0{{$}}
; GCN: buffer_store_byte [[TMP]]
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @zext_bool_icmp_eq_neg1(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp eq i32 %a, %b
  %ext = zext i1 %icmp0 to i32
  %icmp1 = icmp eq i32 %ext, -1
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; Reduces to true:
; FUNC-LABEL: {{^}}zext_bool_icmp_ne_neg1:
; GCN: v_mov_b32_e32 [[TMP:v[0-9]+]], 1{{$}}
; GCN: buffer_store_byte [[TMP]]
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @zext_bool_icmp_ne_neg1(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = zext i1 %icmp0 to i32
  %icmp1 = icmp ne i32 %ext, -1
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}cmp_zext_k_i8max:
; SI: s_load_dword [[VALUE:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dword [[VALUE:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: s_movk_i32 [[K255:s[0-9]+]], 0xff
; SI-DAG: s_and_b32 [[B:s[0-9]+]], [[VALUE]], [[K255]]
; SI: s_cmp_lg_u32 [[B]], [[K255]]
; SI: s_cselect_b64 [[CC:[^,]+]], -1, 0

; VI: v_mov_b32_e32 [[VK255:v[0-9]+]], [[K255]]
; VI: v_and_b32_e32 [[B:v[0-9]+]], [[VALUE]], [[VK255]]
; VI: v_cmp_ne_u16_e32 vcc, [[K255]], [[B]]

; SI: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, [[CC]]
; VI: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, vcc
; GCN: buffer_store_byte [[RESULT]]
; GCN: s_endpgm
define amdgpu_kernel void @cmp_zext_k_i8max(i1 addrspace(1)* %out, i8 %b) nounwind {
  %b.ext = zext i8 %b to i32
  %icmp0 = icmp ne i32 %b.ext, 255
  store i1 %icmp0, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}cmp_sext_k_neg1:
; GCN: buffer_load_sbyte [[B:v[0-9]+]]
; GCN: v_cmp_ne_u32_e32 vcc, -1, [[B]]{{$}}
; GCN-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, vcc
; GCN: buffer_store_byte [[RESULT]]
; GCN: s_endpgm
define amdgpu_kernel void @cmp_sext_k_neg1(i1 addrspace(1)* %out, i8 addrspace(1)* %b.ptr) nounwind {
  %b = load i8, i8 addrspace(1)* %b.ptr
  %b.ext = sext i8 %b to i32
  %icmp0 = icmp ne i32 %b.ext, -1
  store i1 %icmp0, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_cmp_sext_k_neg1_i8_sext_arg:
; GCN: v_cmp_ne_u32_e32 vcc, -1, v0
; GCN: v_cndmask_b32_e64 [[SELECT:v[0-9]+]], 0, 1, vcc
; GCN: buffer_store_byte [[SELECT]]
define void @v_cmp_sext_k_neg1_i8_sext_arg(i8 signext %b) nounwind {
  %b.ext = sext i8 %b to i32
  %icmp0 = icmp ne i32 %b.ext, -1
  store i1 %icmp0, i1 addrspace(1)* undef
  ret void
}

; FIXME: This ends up doing a buffer_load_ubyte, and and compare to
; 255. Seems to be because of ordering problems when not allowing load widths to be reduced.
; Should do a buffer_load_sbyte and compare with -1

; FUNC-LABEL: {{^}}cmp_sext_k_neg1_i8_arg:
; SI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb
; VI: s_load_dword [[VAL:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: s_movk_i32 [[K:s[0-9]+]], 0xff
; GCN-DAG: s_and_b32 [[B:s[0-9]+]], [[VAL]], [[K]]
; GCN: s_cmp_lg_u32 [[B]], [[K]]{{$}}
; GCN: s_cselect_b64 [[CC:[^,]+]], -1, 0
; GCN: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, [[CC]]
; GCN: buffer_store_byte [[RESULT]]
; GCN: s_endpgm
define amdgpu_kernel void @cmp_sext_k_neg1_i8_arg(i1 addrspace(1)* %out, i8 %b) nounwind {
  %b.ext = sext i8 %b to i32
  %icmp0 = icmp ne i32 %b.ext, -1
  store i1 %icmp0, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}cmp_zext_k_neg1:
; GCN: v_mov_b32_e32 [[RESULT:v[0-9]+]], 1{{$}}
; GCN: buffer_store_byte [[RESULT]]
; GCN: s_endpgm
define amdgpu_kernel void @cmp_zext_k_neg1(i1 addrspace(1)* %out, i8 %b) nounwind {
  %b.ext = zext i8 %b to i32
  %icmp0 = icmp ne i32 %b.ext, -1
  store i1 %icmp0, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zext_bool_icmp_ne_k:
; GCN: v_mov_b32_e32 [[RESULT:v[0-9]+]], 1{{$}}
; GCN: buffer_store_byte [[RESULT]]
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @zext_bool_icmp_ne_k(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = zext i1 %icmp0 to i32
  %icmp1 = icmp ne i32 %ext, 2
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zext_bool_icmp_eq_k:
; GCN: v_mov_b32_e32 [[RESULT:v[0-9]+]], 0{{$}}
; GCN: buffer_store_byte [[RESULT]]
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @zext_bool_icmp_eq_k(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = zext i1 %icmp0 to i32
  %icmp1 = icmp eq i32 %ext, 2
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FIXME: These cases should really be able fold to true/false in
; DAGCombiner

; This really folds away to false
; FUNC-LABEL: {{^}}sext_bool_icmp_eq_1:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 0{{$}}
; GCN: buffer_store_byte [[K]]
define amdgpu_kernel void @sext_bool_icmp_eq_1(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp eq i32 %a, %b
  %ext = sext i1 %icmp0 to i32
  %icmp1 = icmp eq i32 %ext, 1
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sext_bool_icmp_ne_1:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 1{{$}}
; GCN: buffer_store_byte [[K]]
define amdgpu_kernel void @sext_bool_icmp_ne_1(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = sext i1 %icmp0 to i32
  %icmp1 = icmp ne i32 %ext, 1
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sext_bool_icmp_ne_k:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 1{{$}}
; GCN: buffer_store_byte [[K]]
define amdgpu_kernel void @sext_bool_icmp_ne_k(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = sext i1 %icmp0 to i32
  %icmp1 = icmp ne i32 %ext, 2
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

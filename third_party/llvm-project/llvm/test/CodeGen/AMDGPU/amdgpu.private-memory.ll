; RUN: llc -show-mc-encoding --amdhsa-code-object-version=2 -mattr=+promote-alloca -disable-promote-alloca-to-vector -amdgpu-load-store-vectorizer=0 -enable-amdgpu-aa=0 -verify-machineinstrs -march=amdgcn < %s | FileCheck -enable-var-scope -check-prefix=SI-PROMOTE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -show-mc-encoding --amdhsa-code-object-version=2 -mattr=+promote-alloca -disable-promote-alloca-to-vector -amdgpu-load-store-vectorizer=0 -enable-amdgpu-aa=0 -verify-machineinstrs -mtriple=amdgcn--amdhsa -mcpu=kaveri -mattr=-unaligned-access-mode < %s | FileCheck -enable-var-scope -check-prefix=SI-PROMOTE -check-prefix=SI -check-prefix=FUNC -check-prefix=HSA-PROMOTE %s
; RUN: llc -show-mc-encoding --amdhsa-code-object-version=2 -mattr=-promote-alloca -amdgpu-load-store-vectorizer=0 -enable-amdgpu-aa=0 -verify-machineinstrs -march=amdgcn < %s | FileCheck %s -check-prefix=SI-ALLOCA -check-prefix=SI -check-prefix=FUNC
; RUN: llc -show-mc-encoding --amdhsa-code-object-version=2 -mattr=-promote-alloca -amdgpu-load-store-vectorizer=0 -enable-amdgpu-aa=0 -verify-machineinstrs -mtriple=amdgcn-amdhsa -mcpu=kaveri -mattr=-unaligned-access-mode < %s | FileCheck -enable-var-scope -check-prefix=SI-ALLOCA -check-prefix=SI -check-prefix=FUNC -check-prefix=HSA-ALLOCA %s
; RUN: llc -show-mc-encoding --amdhsa-code-object-version=2 -mattr=+promote-alloca -disable-promote-alloca-to-vector -amdgpu-load-store-vectorizer=0 -enable-amdgpu-aa=0 -verify-machineinstrs -mtriple=amdgcn-amdhsa -march=amdgcn -mcpu=tonga -mattr=-unaligned-access-mode < %s | FileCheck -enable-var-scope -check-prefix=SI-PROMOTE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -show-mc-encoding --amdhsa-code-object-version=2 -mattr=+promote-alloca -amdgpu-load-store-vectorizer=0 -enable-amdgpu-aa=0 -verify-machineinstrs -mtriple=amdgcn-amdhsa -march=amdgcn -mcpu=tonga -mattr=-unaligned-access-mode < %s | FileCheck -enable-var-scope -check-prefix=SI-PROMOTE-VECT -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -show-mc-encoding --amdhsa-code-object-version=2 -mattr=-promote-alloca -amdgpu-load-store-vectorizer=0 -enable-amdgpu-aa=0 -verify-machineinstrs -mtriple=amdgcn-amdhsa -march=amdgcn -mcpu=tonga -mattr=-unaligned-access-mode < %s | FileCheck -enable-var-scope -check-prefix=SI-ALLOCA -check-prefix=SI -check-prefix=FUNC %s

; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -data-layout=A5 -mcpu=kaveri -amdgpu-promote-alloca -disable-promote-alloca-to-vector < %s | FileCheck -enable-var-scope -check-prefix=HSAOPT -check-prefix=OPT %s
; RUN: opt -S -mtriple=amdgcn-unknown-unknown -data-layout=A5 -mcpu=kaveri -amdgpu-promote-alloca -disable-promote-alloca-to-vector < %s | FileCheck -enable-var-scope -check-prefix=NOHSAOPT -check-prefix=OPT %s

; RUN: llc -march=r600 -mcpu=cypress -disable-promote-alloca-to-vector < %s | FileCheck %s -check-prefix=R600 -check-prefix=FUNC
; RUN: llc -march=r600 -mcpu=cypress < %s | FileCheck %s -check-prefix=R600-VECT -check-prefix=FUNC

; HSAOPT: @mova_same_clause.stack = internal unnamed_addr addrspace(3) global [256 x [5 x i32]] undef, align 4
; HSAOPT: @high_alignment.stack = internal unnamed_addr addrspace(3) global [256 x [8 x i32]] undef, align 16


; FUNC-LABEL: {{^}}mova_same_clause:
; OPT-LABEL: @mova_same_clause(

; R600: LDS_WRITE
; R600: LDS_WRITE
; R600: LDS_READ
; R600: LDS_READ

; HSA-PROMOTE: .amd_kernel_code_t
; HSA-PROMOTE: workgroup_group_segment_byte_size = 5120
; HSA-PROMOTE: .end_amd_kernel_code_t

; HSA-PROMOTE: s_load_dword s{{[0-9]+}}, s[4:5], 0x2

; SI-PROMOTE: ds_write_b32
; SI-PROMOTE: ds_write_b32
; SI-PROMOTE: ds_read_b32
; SI-PROMOTE: ds_read_b32

; HSA-ALLOCA: .amd_kernel_code_t
; FIXME: Creating the emergency stack slots causes us to over-estimate scratch
; by 4 bytes.
; HSA-ALLOCA: workitem_private_segment_byte_size = 24
; HSA-ALLOCA: .end_amd_kernel_code_t

; HSA-ALLOCA: s_mov_b32 flat_scratch_lo, s7
; HSA-ALLOCA: s_add_i32 s6, s6, s9
; HSA-ALLOCA: s_lshr_b32 flat_scratch_hi, s6, 8

; SI-ALLOCA: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen ; encoding: [0x00,0x10,0x70,0xe0
; SI-ALLOCA: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen ; encoding: [0x00,0x10,0x70,0xe0


; HSAOPT: [[DISPATCH_PTR:%[0-9]+]] = call noalias nonnull dereferenceable(64) i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
; HSAOPT: [[CAST_DISPATCH_PTR:%[0-9]+]] = bitcast i8 addrspace(4)* [[DISPATCH_PTR]] to i32 addrspace(4)*
; HSAOPT: [[GEP0:%[0-9]+]] = getelementptr inbounds i32, i32 addrspace(4)* [[CAST_DISPATCH_PTR]], i64 1
; HSAOPT: [[LDXY:%[0-9]+]] = load i32, i32 addrspace(4)* [[GEP0]], align 4, !invariant.load !0
; HSAOPT: [[GEP1:%[0-9]+]] = getelementptr inbounds i32, i32 addrspace(4)* [[CAST_DISPATCH_PTR]], i64 2
; HSAOPT: [[LDZU:%[0-9]+]] = load i32, i32 addrspace(4)* [[GEP1]], align 4, !range !1, !invariant.load !0
; HSAOPT: [[EXTRACTY:%[0-9]+]] = lshr i32 [[LDXY]], 16

; HSAOPT: [[WORKITEM_ID_X:%[0-9]+]] = call i32 @llvm.amdgcn.workitem.id.x(), !range !2
; HSAOPT: [[WORKITEM_ID_Y:%[0-9]+]] = call i32 @llvm.amdgcn.workitem.id.y(), !range !2
; HSAOPT: [[WORKITEM_ID_Z:%[0-9]+]] = call i32 @llvm.amdgcn.workitem.id.z(), !range !2

; HSAOPT: [[Y_SIZE_X_Z_SIZE:%[0-9]+]] = mul nuw nsw i32 [[EXTRACTY]], [[LDZU]]
; HSAOPT: [[YZ_X_XID:%[0-9]+]] = mul i32 [[Y_SIZE_X_Z_SIZE]], [[WORKITEM_ID_X]]
; HSAOPT: [[Y_X_Z_SIZE:%[0-9]+]] = mul nuw nsw i32 [[WORKITEM_ID_Y]], [[LDZU]]
; HSAOPT: [[ADD_YZ_X_X_YZ_SIZE:%[0-9]+]] = add i32 [[YZ_X_XID]], [[Y_X_Z_SIZE]]
; HSAOPT: [[ADD_ZID:%[0-9]+]] = add i32 [[ADD_YZ_X_X_YZ_SIZE]], [[WORKITEM_ID_Z]]

; HSAOPT: [[LOCAL_GEP:%[0-9]+]] = getelementptr inbounds [256 x [5 x i32]], [256 x [5 x i32]] addrspace(3)* @mova_same_clause.stack, i32 0, i32 [[ADD_ZID]]
; HSAOPT: %arrayidx1 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(3)* [[LOCAL_GEP]], i32 0, i32 {{%[0-9]+}}
; HSAOPT: %arrayidx3 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(3)* [[LOCAL_GEP]], i32 0, i32 {{%[0-9]+}}
; HSAOPT: %arrayidx10 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(3)* [[LOCAL_GEP]], i32 0, i32 0
; HSAOPT: %arrayidx12 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(3)* [[LOCAL_GEP]], i32 0, i32 1


; NOHSAOPT: call i32 @llvm.r600.read.local.size.y(), !range !0
; NOHSAOPT: call i32 @llvm.r600.read.local.size.z(), !range !0
; NOHSAOPT: call i32 @llvm.amdgcn.workitem.id.x(), !range !1
; NOHSAOPT: call i32 @llvm.amdgcn.workitem.id.y(), !range !1
; NOHSAOPT: call i32 @llvm.amdgcn.workitem.id.z(), !range !1
define amdgpu_kernel void @mova_same_clause(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) #0 {
entry:
  %stack = alloca [5 x i32], align 4, addrspace(5)
  %0 = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 %0
  store i32 4, i32 addrspace(5)* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 1
  %1 = load i32, i32 addrspace(1)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 %1
  store i32 5, i32 addrspace(5)* %arrayidx3, align 4
  %arrayidx10 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 0
  %2 = load i32, i32 addrspace(5)* %arrayidx10, align 4
  store i32 %2, i32 addrspace(1)* %out, align 4
  %arrayidx12 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 1
  %3 = load i32, i32 addrspace(5)* %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 1
  store i32 %3, i32 addrspace(1)* %arrayidx13
  ret void
}

; OPT-LABEL: @high_alignment(
; OPT: getelementptr inbounds [256 x [8 x i32]], [256 x [8 x i32]] addrspace(3)* @high_alignment.stack, i32 0, i32 %{{[0-9]+}}
define amdgpu_kernel void @high_alignment(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) #0 {
entry:
  %stack = alloca [8 x i32], align 16, addrspace(5)
  %0 = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [8 x i32], [8 x i32] addrspace(5)* %stack, i32 0, i32 %0
  store i32 4, i32 addrspace(5)* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 1
  %1 = load i32, i32 addrspace(1)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [8 x i32], [8 x i32] addrspace(5)* %stack, i32 0, i32 %1
  store i32 5, i32 addrspace(5)* %arrayidx3, align 4
  %arrayidx10 = getelementptr inbounds [8 x i32], [8 x i32] addrspace(5)* %stack, i32 0, i32 0
  %2 = load i32, i32 addrspace(5)* %arrayidx10, align 4
  store i32 %2, i32 addrspace(1)* %out, align 4
  %arrayidx12 = getelementptr inbounds [8 x i32], [8 x i32] addrspace(5)* %stack, i32 0, i32 1
  %3 = load i32, i32 addrspace(5)* %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 1
  store i32 %3, i32 addrspace(1)* %arrayidx13
  ret void
}

; FUNC-LABEL: {{^}}no_replace_inbounds_gep:
; OPT-LABEL: @no_replace_inbounds_gep(
; OPT: alloca [5 x i32]

; SI-NOT: ds_write
define amdgpu_kernel void @no_replace_inbounds_gep(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) #0 {
entry:
  %stack = alloca [5 x i32], align 4, addrspace(5)
  %0 = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 %0
  store i32 4, i32 addrspace(5)* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 1
  %1 = load i32, i32 addrspace(1)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 %1
  store i32 5, i32 addrspace(5)* %arrayidx3, align 4
  %arrayidx10 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 0
  %2 = load i32, i32 addrspace(5)* %arrayidx10, align 4
  store i32 %2, i32 addrspace(1)* %out, align 4
  %arrayidx12 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 1
  %3 = load i32, i32 addrspace(5)* %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 1
  store i32 %3, i32 addrspace(1)* %arrayidx13
  ret void
}

; This test checks that the stack offset is calculated correctly for structs.
; All register loads/stores should be optimized away, so there shouldn't be
; any MOVA instructions.
;
; XXX: This generated code has unnecessary MOVs, we should be able to optimize
; this.

; FUNC-LABEL: {{^}}multiple_structs:
; OPT-LABEL: @multiple_structs(

; R600-NOT: MOVA_INT
; SI-NOT: v_movrel
; SI-NOT: v_movrel
%struct.point = type { i32, i32 }

define amdgpu_kernel void @multiple_structs(i32 addrspace(1)* %out) #0 {
entry:
  %a = alloca %struct.point, addrspace(5)
  %b = alloca %struct.point, addrspace(5)
  %a.x.ptr = getelementptr %struct.point, %struct.point addrspace(5)* %a, i32 0, i32 0
  %a.y.ptr = getelementptr %struct.point, %struct.point addrspace(5)* %a, i32 0, i32 1
  %b.x.ptr = getelementptr %struct.point, %struct.point addrspace(5)* %b, i32 0, i32 0
  %b.y.ptr = getelementptr %struct.point, %struct.point addrspace(5)* %b, i32 0, i32 1
  store i32 0, i32 addrspace(5)* %a.x.ptr
  store i32 1, i32 addrspace(5)* %a.y.ptr
  store i32 2, i32 addrspace(5)* %b.x.ptr
  store i32 3, i32 addrspace(5)* %b.y.ptr
  %a.indirect.ptr = getelementptr %struct.point, %struct.point addrspace(5)* %a, i32 0, i32 0
  %b.indirect.ptr = getelementptr %struct.point, %struct.point addrspace(5)* %b, i32 0, i32 0
  %a.indirect = load i32, i32 addrspace(5)* %a.indirect.ptr
  %b.indirect = load i32, i32 addrspace(5)* %b.indirect.ptr
  %0 = add i32 %a.indirect, %b.indirect
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; Test direct access of a private array inside a loop.  The private array
; loads and stores should be lowered to copies, so there shouldn't be any
; MOVA instructions.

; FUNC-LABEL: {{^}}direct_loop:
; R600-NOT: MOVA_INT
; SI-NOT: v_movrel

define amdgpu_kernel void @direct_loop(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
entry:
  %prv_array_const = alloca [2 x i32], addrspace(5)
  %prv_array = alloca [2 x i32], addrspace(5)
  %a = load i32, i32 addrspace(1)* %in
  %b_src_ptr = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 1
  %b = load i32, i32 addrspace(1)* %b_src_ptr
  %a_dst_ptr = getelementptr inbounds [2 x i32], [2 x i32] addrspace(5)* %prv_array_const, i32 0, i32 0
  store i32 %a, i32 addrspace(5)* %a_dst_ptr
  %b_dst_ptr = getelementptr inbounds [2 x i32], [2 x i32] addrspace(5)* %prv_array_const, i32 0, i32 1
  store i32 %b, i32 addrspace(5)* %b_dst_ptr
  br label %for.body

for.body:
  %inc = phi i32 [0, %entry], [%count, %for.body]
  %x_ptr = getelementptr inbounds [2 x i32], [2 x i32] addrspace(5)* %prv_array_const, i32 0, i32 0
  %x = load i32, i32 addrspace(5)* %x_ptr
  %y_ptr = getelementptr inbounds [2 x i32], [2 x i32] addrspace(5)* %prv_array, i32 0, i32 0
  %y = load i32, i32 addrspace(5)* %y_ptr
  %xy = add i32 %x, %y
  store i32 %xy, i32 addrspace(5)* %y_ptr
  %count = add i32 %inc, 1
  %done = icmp eq i32 %count, 4095
  br i1 %done, label %for.end, label %for.body

for.end:
  %value_ptr = getelementptr inbounds [2 x i32], [2 x i32] addrspace(5)* %prv_array, i32 0, i32 0
  %value = load i32, i32 addrspace(5)* %value_ptr
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}short_array:

; R600-VECT: MOVA_INT

; SI-ALLOCA-DAG: buffer_store_short v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0 offset:6 ; encoding: [0x06,0x00,0x68,0xe0
; SI-ALLOCA-DAG: buffer_store_short v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0 offset:4 ; encoding: [0x04,0x00,0x68,0xe0
; Loaded value is 0 or 1, so sext will become zext, so we get buffer_load_ushort instead of buffer_load_sshort.
; SI-ALLOCA: buffer_load_sshort v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0

; SI-PROMOTE-VECT: s_load_dword [[IDX:s[0-9]+]]
; SI-PROMOTE-VECT: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 4
; SI-PROMOTE-VECT: s_lshr_b32 [[SREG:s[0-9]+]], 0x10000, [[SCALED_IDX]]
; SI-PROMOTE-VECT: s_and_b32 s{{[0-9]+}}, [[SREG]], 0xffff
define amdgpu_kernel void @short_array(i32 addrspace(1)* %out, i32 %index) #0 {
entry:
  %0 = alloca [2 x i16], addrspace(5)
  %1 = getelementptr inbounds [2 x i16], [2 x i16] addrspace(5)* %0, i32 0, i32 0
  %2 = getelementptr inbounds [2 x i16], [2 x i16] addrspace(5)* %0, i32 0, i32 1
  store i16 0, i16 addrspace(5)* %1
  store i16 1, i16 addrspace(5)* %2
  %3 = getelementptr inbounds [2 x i16], [2 x i16] addrspace(5)* %0, i32 0, i32 %index
  %4 = load i16, i16 addrspace(5)* %3
  %5 = sext i16 %4 to i32
  store i32 %5, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}char_array:

; R600-VECT: MOVA_INT

; SI-PROMOTE-VECT-DAG: s_lshl_b32
; SI-PROMOTE-VECT-DAG: v_lshrrev

; SI-ALLOCA-DAG: buffer_store_byte v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0 offset:4 ; encoding: [0x04,0x00,0x60,0xe0
; SI-ALLOCA-DAG: buffer_store_byte v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0 offset:5 ; encoding: [0x05,0x00,0x60,0xe0
define amdgpu_kernel void @char_array(i32 addrspace(1)* %out, i32 %index) #0 {
entry:
  %0 = alloca [2 x i8], addrspace(5)
  %1 = getelementptr inbounds [2 x i8], [2 x i8] addrspace(5)* %0, i32 0, i32 0
  %2 = getelementptr inbounds [2 x i8], [2 x i8] addrspace(5)* %0, i32 0, i32 1
  store i8 0, i8 addrspace(5)* %1
  store i8 1, i8 addrspace(5)* %2
  %3 = getelementptr inbounds [2 x i8], [2 x i8] addrspace(5)* %0, i32 0, i32 %index
  %4 = load i8, i8 addrspace(5)* %3
  %5 = sext i8 %4 to i32
  store i32 %5, i32 addrspace(1)* %out
  ret void
}

; Test that two stack objects are not stored in the same register
; The second stack object should be in T3.X
; FUNC-LABEL: {{^}}no_overlap:
;
; A total of 5 bytes should be allocated and used.
; SI: buffer_store_byte v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0 offset:4 ;
define amdgpu_kernel void @no_overlap(i32 addrspace(1)* %out, i32 %in) #0 {
entry:
  %0 = alloca [3 x i8], align 1, addrspace(5)
  %1 = alloca [2 x i8], align 1, addrspace(5)
  %2 = getelementptr [3 x i8], [3 x i8] addrspace(5)* %0, i32 0, i32 0
  %3 = getelementptr [3 x i8], [3 x i8] addrspace(5)* %0, i32 0, i32 1
  %4 = getelementptr [3 x i8], [3 x i8] addrspace(5)* %0, i32 0, i32 2
  %5 = getelementptr [2 x i8], [2 x i8] addrspace(5)* %1, i32 0, i32 0
  %6 = getelementptr [2 x i8], [2 x i8] addrspace(5)* %1, i32 0, i32 1
  store i8 0, i8 addrspace(5)* %2
  store i8 1, i8 addrspace(5)* %3
  store i8 2, i8 addrspace(5)* %4
  store i8 1, i8 addrspace(5)* %5
  store i8 0, i8 addrspace(5)* %6
  %7 = getelementptr [3 x i8], [3 x i8] addrspace(5)* %0, i32 0, i32 %in
  %8 = getelementptr [2 x i8], [2 x i8] addrspace(5)* %1, i32 0, i32 %in
  %9 = load i8, i8 addrspace(5)* %7
  %10 = load i8, i8 addrspace(5)* %8
  %11 = add i8 %9, %10
  %12 = sext i8 %11 to i32
  store i32 %12, i32 addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @char_array_array(i32 addrspace(1)* %out, i32 %index) #0 {
entry:
  %alloca = alloca [2 x [2 x i8]], addrspace(5)
  %gep0 = getelementptr [2 x [2 x i8]], [2 x [2 x i8]] addrspace(5)* %alloca, i32 0, i32 0, i32 0
  %gep1 = getelementptr [2 x [2 x i8]], [2 x [2 x i8]] addrspace(5)* %alloca, i32 0, i32 0, i32 1
  store i8 0, i8 addrspace(5)* %gep0
  store i8 1, i8 addrspace(5)* %gep1
  %gep2 = getelementptr [2 x [2 x i8]], [2 x [2 x i8]] addrspace(5)* %alloca, i32 0, i32 0, i32 %index
  %load = load i8, i8 addrspace(5)* %gep2
  %sext = sext i8 %load to i32
  store i32 %sext, i32 addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @i32_array_array(i32 addrspace(1)* %out, i32 %index) #0 {
entry:
  %alloca = alloca [2 x [2 x i32]], addrspace(5)
  %gep0 = getelementptr [2 x [2 x i32]], [2 x [2 x i32]] addrspace(5)* %alloca, i32 0, i32 0, i32 0
  %gep1 = getelementptr [2 x [2 x i32]], [2 x [2 x i32]] addrspace(5)* %alloca, i32 0, i32 0, i32 1
  store i32 0, i32 addrspace(5)* %gep0
  store i32 1, i32 addrspace(5)* %gep1
  %gep2 = getelementptr [2 x [2 x i32]], [2 x [2 x i32]] addrspace(5)* %alloca, i32 0, i32 0, i32 %index
  %load = load i32, i32 addrspace(5)* %gep2
  store i32 %load, i32 addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @i64_array_array(i64 addrspace(1)* %out, i32 %index) #0 {
entry:
  %alloca = alloca [2 x [2 x i64]], addrspace(5)
  %gep0 = getelementptr [2 x [2 x i64]], [2 x [2 x i64]] addrspace(5)* %alloca, i32 0, i32 0, i32 0
  %gep1 = getelementptr [2 x [2 x i64]], [2 x [2 x i64]] addrspace(5)* %alloca, i32 0, i32 0, i32 1
  store i64 0, i64 addrspace(5)* %gep0
  store i64 1, i64 addrspace(5)* %gep1
  %gep2 = getelementptr [2 x [2 x i64]], [2 x [2 x i64]] addrspace(5)* %alloca, i32 0, i32 0, i32 %index
  %load = load i64, i64 addrspace(5)* %gep2
  store i64 %load, i64 addrspace(1)* %out
  ret void
}

%struct.pair32 = type { i32, i32 }

define amdgpu_kernel void @struct_array_array(i32 addrspace(1)* %out, i32 %index) #0 {
entry:
  %alloca = alloca [2 x [2 x %struct.pair32]], addrspace(5)
  %gep0 = getelementptr [2 x [2 x %struct.pair32]], [2 x [2 x %struct.pair32]] addrspace(5)* %alloca, i32 0, i32 0, i32 0, i32 1
  %gep1 = getelementptr [2 x [2 x %struct.pair32]], [2 x [2 x %struct.pair32]] addrspace(5)* %alloca, i32 0, i32 0, i32 1, i32 1
  store i32 0, i32 addrspace(5)* %gep0
  store i32 1, i32 addrspace(5)* %gep1
  %gep2 = getelementptr [2 x [2 x %struct.pair32]], [2 x [2 x %struct.pair32]] addrspace(5)* %alloca, i32 0, i32 0, i32 %index, i32 0
  %load = load i32, i32 addrspace(5)* %gep2
  store i32 %load, i32 addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @struct_pair32_array(i32 addrspace(1)* %out, i32 %index) #0 {
entry:
  %alloca = alloca [2 x %struct.pair32], addrspace(5)
  %gep0 = getelementptr [2 x %struct.pair32], [2 x %struct.pair32] addrspace(5)* %alloca, i32 0, i32 0, i32 1
  %gep1 = getelementptr [2 x %struct.pair32], [2 x %struct.pair32] addrspace(5)* %alloca, i32 0, i32 1, i32 0
  store i32 0, i32 addrspace(5)* %gep0
  store i32 1, i32 addrspace(5)* %gep1
  %gep2 = getelementptr [2 x %struct.pair32], [2 x %struct.pair32] addrspace(5)* %alloca, i32 0, i32 %index, i32 0
  %load = load i32, i32 addrspace(5)* %gep2
  store i32 %load, i32 addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @select_private(i32 addrspace(1)* %out, i32 %in) nounwind {
entry:
  %tmp = alloca [2 x i32], addrspace(5)
  %tmp1 = getelementptr [2 x i32], [2 x i32] addrspace(5)* %tmp, i32 0, i32 0
  %tmp2 = getelementptr [2 x i32], [2 x i32] addrspace(5)* %tmp, i32 0, i32 1
  store i32 0, i32 addrspace(5)* %tmp1
  store i32 1, i32 addrspace(5)* %tmp2
  %cmp = icmp eq i32 %in, 0
  %sel = select i1 %cmp, i32 addrspace(5)* %tmp1, i32 addrspace(5)* %tmp2
  %load = load i32, i32 addrspace(5)* %sel
  store i32 %load, i32 addrspace(1)* %out
  ret void
}

; AMDGPUPromoteAlloca does not know how to handle ptrtoint.  When it
; finds one, it should stop trying to promote.

; FUNC-LABEL: ptrtoint:
; SI-NOT: ds_write
; SI: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen
; SI: v_add_{{[iu]}}32_e32 [[ADD_OFFSET:v[0-9]+]], vcc, 5,
; SI: buffer_load_dword v{{[0-9]+}}, [[ADD_OFFSET:v[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0 offen ;
define amdgpu_kernel void @ptrtoint(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %alloca = alloca [16 x i32], addrspace(5)
  %tmp0 = getelementptr [16 x i32], [16 x i32] addrspace(5)* %alloca, i32 0, i32 %a
  store i32 5, i32 addrspace(5)* %tmp0
  %tmp1 = ptrtoint [16 x i32] addrspace(5)* %alloca to i32
  %tmp2 = add i32 %tmp1, 5
  %tmp3 = inttoptr i32 %tmp2 to i32 addrspace(5)*
  %tmp4 = getelementptr i32, i32 addrspace(5)* %tmp3, i32 %b
  %tmp5 = load i32, i32 addrspace(5)* %tmp4
  store i32 %tmp5, i32 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @pointer_typed_alloca(
; OPT:  getelementptr inbounds [256 x i32 addrspace(1)*], [256 x i32 addrspace(1)*] addrspace(3)* @pointer_typed_alloca.A.addr, i32 0, i32 %{{[0-9]+}}
; OPT: load i32 addrspace(1)*, i32 addrspace(1)* addrspace(3)* %{{[0-9]+}}, align 4
define amdgpu_kernel void @pointer_typed_alloca(i32 addrspace(1)* %A) #1 {
entry:
  %A.addr = alloca i32 addrspace(1)*, align 4, addrspace(5)
  store i32 addrspace(1)* %A, i32 addrspace(1)* addrspace(5)* %A.addr, align 4
  %ld0 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(5)* %A.addr, align 4
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %ld0, i32 0
  store i32 1, i32 addrspace(1)* %arrayidx, align 4
  %ld1 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(5)* %A.addr, align 4
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %ld1, i32 1
  store i32 2, i32 addrspace(1)* %arrayidx1, align 4
  %ld2 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(5)* %A.addr, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %ld2, i32 2
  store i32 3, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}

; FUNC-LABEL: v16i32_stack:

; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT

; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword

define amdgpu_kernel void @v16i32_stack(<16 x i32> addrspace(1)* %out, i32 %a) {
  %alloca = alloca [2 x <16 x i32>], addrspace(5)
  %tmp0 = getelementptr [2 x <16 x i32>], [2 x <16 x i32>] addrspace(5)* %alloca, i32 0, i32 %a
  %tmp5 = load <16 x i32>, <16 x i32> addrspace(5)* %tmp0
  store <16 x i32> %tmp5, <16 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: v16float_stack:

; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT
; R600: MOVA_INT

; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword
; SI: buffer_load_dword

define amdgpu_kernel void @v16float_stack(<16 x float> addrspace(1)* %out, i32 %a) {
  %alloca = alloca [2 x <16 x float>], addrspace(5)
  %tmp0 = getelementptr [2 x <16 x float>], [2 x <16 x float>] addrspace(5)* %alloca, i32 0, i32 %a
  %tmp5 = load <16 x float>, <16 x float> addrspace(5)* %tmp0
  store <16 x float> %tmp5, <16 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: v2float_stack:

; R600: MOVA_INT
; R600: MOVA_INT

; SI: buffer_load_dword
; SI: buffer_load_dword

define amdgpu_kernel void @v2float_stack(<2 x float> addrspace(1)* %out, i32 %a) {
  %alloca = alloca [16 x <2 x float>], addrspace(5)
  %tmp0 = getelementptr [16 x <2 x float>], [16 x <2 x float>] addrspace(5)* %alloca, i32 0, i32 %a
  %tmp5 = load <2 x float>, <2 x float> addrspace(5)* %tmp0
  store <2 x float> %tmp5, <2 x float> addrspace(1)* %out
  ret void
}

; OPT-LABEL: @direct_alloca_read_0xi32(
; OPT: store [0 x i32] undef, [0 x i32] addrspace(3)*
; OPT: load [0 x i32], [0 x i32] addrspace(3)*
define amdgpu_kernel void @direct_alloca_read_0xi32([0 x i32] addrspace(1)* %out, i32 %index) {
entry:
  %tmp = alloca [0 x i32], addrspace(5)
  store [0 x i32] [], [0 x i32] addrspace(5)* %tmp
  %load = load [0 x i32], [0 x i32] addrspace(5)* %tmp
  store [0 x i32] %load, [0 x i32] addrspace(1)* %out
  ret void
}

; OPT-LABEL: @direct_alloca_read_1xi32(
; OPT: store [1 x i32] zeroinitializer, [1 x i32] addrspace(3)*
; OPT: load [1 x i32], [1 x i32] addrspace(3)*
define amdgpu_kernel void @direct_alloca_read_1xi32([1 x i32] addrspace(1)* %out, i32 %index) {
entry:
  %tmp = alloca [1 x i32], addrspace(5)
  store [1 x i32] [i32 0], [1 x i32] addrspace(5)* %tmp
  %load = load [1 x i32], [1 x i32] addrspace(5)* %tmp
  store [1 x i32] %load, [1 x i32] addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind "amdgpu-waves-per-eu"="1,2" "amdgpu-flat-work-group-size"="1,256" }
attributes #1 = { nounwind "amdgpu-flat-work-group-size"="1,256" }

; HSAOPT: !0 = !{}
; HSAOPT: !1 = !{i32 0, i32 257}
; HSAOPT: !2 = !{i32 0, i32 256}

; NOHSAOPT: !0 = !{i32 0, i32 257}
; NOHSAOPT: !1 = !{i32 0, i32 256}

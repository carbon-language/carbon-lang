; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=GFX9 %s

; GCN-LABEL: {{^}}fma_vector_vector_scalar_lo:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[SCALAR0]] op_sel_hi:[1,1,0]{{$}}
define amdgpu_kernel void @fma_vector_vector_scalar_lo(<2 x half> addrspace(1)* %out, <2 x half> addrspace(3)* %lds, half addrspace(3)* %arg2) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, <2 x half> addrspace(3)* %lds, i32 1

  %vec0 = load volatile <2 x half>, <2 x half> addrspace(3)* %lds, align 4
  %vec1 = load volatile <2 x half>, <2 x half> addrspace(3)* %lds.gep1, align 4
  %scalar0 = load volatile half, half addrspace(3)* %arg2, align 2

  %scalar0.vec = insertelement <2 x half> undef, half %scalar0, i32 0
  %scalar0.broadcast = shufflevector <2 x half> %scalar0.vec, <2 x half> undef, <2 x i32> zeroinitializer

  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %scalar0.broadcast)
  store <2 x half> %result, <2 x half> addrspace(1)* %out, align 4
  ret void
}

; Apply fneg to broadcasted vector
; GCN-LABEL: {{^}}fma_vector_vector_neg_broadcast_scalar_lo:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[SCALAR0]] op_sel_hi:[1,1,0] neg_lo:[0,0,1] neg_hi:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_neg_broadcast_scalar_lo(<2 x half> addrspace(1)* %out, <2 x half> addrspace(3)* %lds, half addrspace(3)* %arg2) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, <2 x half> addrspace(3)* %lds, i32 1

  %vec0 = load volatile <2 x half>, <2 x half> addrspace(3)* %lds, align 4
  %vec1 = load volatile <2 x half>, <2 x half> addrspace(3)* %lds.gep1, align 4
  %scalar0 = load volatile half, half addrspace(3)* %arg2, align 2

  %scalar0.vec = insertelement <2 x half> undef, half %scalar0, i32 0
  %scalar0.broadcast = shufflevector <2 x half> %scalar0.vec, <2 x half> undef, <2 x i32> zeroinitializer
  %neg.scalar0.broadcast = fsub <2 x half> <half -0.0, half -0.0>, %scalar0.broadcast

  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %neg.scalar0.broadcast)
  store <2 x half> %result, <2 x half> addrspace(1)* %out, align 4
  ret void
}

; Apply fneg before broadcast
; GCN-LABEL: {{^}}fma_vector_vector_neg_scalar_lo:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[SCALAR0]] op_sel_hi:[1,1,0] neg_lo:[0,0,1] neg_hi:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_neg_scalar_lo(<2 x half> addrspace(1)* %out, <2 x half> addrspace(3)* %lds, half addrspace(3)* %arg2) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, <2 x half> addrspace(3)* %lds, i32 1

  %vec0 = load volatile <2 x half>, <2 x half> addrspace(3)* %lds, align 4
  %vec1 = load volatile <2 x half>, <2 x half> addrspace(3)* %lds.gep1, align 4
  %scalar0 = load volatile half, half addrspace(3)* %arg2, align 2

  %neg.scalar0 = fsub half -0.0, %scalar0
  %neg.scalar0.vec = insertelement <2 x half> undef, half %neg.scalar0, i32 0
  %neg.scalar0.broadcast = shufflevector <2 x half> %neg.scalar0.vec, <2 x half> undef, <2 x i32> zeroinitializer

  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %neg.scalar0.broadcast)
  store <2 x half> %result, <2 x half> addrspace(1)* %out, align 4
  ret void
}

; Apply fneg before and after broadcast, and should cancel out.
; GCN-LABEL: {{^}}fma_vector_vector_neg_broadcast_neg_scalar_lo:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[SCALAR0]] op_sel_hi:[1,1,0]{{$}}
define amdgpu_kernel void @fma_vector_vector_neg_broadcast_neg_scalar_lo(<2 x half> addrspace(1)* %out, <2 x half> addrspace(3)* %lds, half addrspace(3)* %arg2) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, <2 x half> addrspace(3)* %lds, i32 1

  %vec0 = load volatile <2 x half>, <2 x half> addrspace(3)* %lds, align 4
  %vec1 = load volatile <2 x half>, <2 x half> addrspace(3)* %lds.gep1, align 4
  %scalar0 = load volatile half, half addrspace(3)* %arg2, align 2

  %neg.scalar0 = fsub half -0.0, %scalar0
  %neg.scalar0.vec = insertelement <2 x half> undef, half %neg.scalar0, i32 0
  %neg.scalar0.broadcast = shufflevector <2 x half> %neg.scalar0.vec, <2 x half> undef, <2 x i32> zeroinitializer
  %neg.neg.scalar0.broadcast = fsub <2 x half> <half -0.0, half -0.0>, %neg.scalar0.broadcast

  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %neg.neg.scalar0.broadcast)
  store <2 x half> %result, <2 x half> addrspace(1)* %out, align 4
  ret void
}

; Add scalar, but negate low component
; GCN-LABEL: {{^}}fma_vector_vector_scalar_neg_lo:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[SCALAR0]] op_sel_hi:[1,1,0] neg_lo:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_scalar_neg_lo(<2 x half> addrspace(1)* %out, <2 x half> addrspace(3)* %lds, half addrspace(3)* %arg2) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, <2 x half> addrspace(3)* %lds, i32 1

  %vec0 = load volatile <2 x half>, <2 x half> addrspace(3)* %lds, align 4
  %vec1 = load volatile <2 x half>, <2 x half> addrspace(3)* %lds.gep1, align 4
  %scalar0 = load volatile half, half addrspace(3)* %arg2, align 2

  %neg.scalar0 = fsub half -0.0, %scalar0
  %neg.scalar0.vec = insertelement <2 x half> undef, half %neg.scalar0, i32 0
  %neg.scalar0.scalar0 = insertelement <2 x half> %neg.scalar0.vec, half %scalar0, i32 1
  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %neg.scalar0.scalar0)
  store <2 x half> %result, <2 x half> addrspace(1)* %out, align 4
  ret void
}

; Add scalar, but negate high component
; GCN-LABEL: {{^}}fma_vector_vector_scalar_neg_hi:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[SCALAR0]] op_sel_hi:[1,1,0] neg_hi:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_scalar_neg_hi(<2 x half> addrspace(1)* %out, <2 x half> addrspace(3)* %lds, half addrspace(3)* %arg2) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, <2 x half> addrspace(3)* %lds, i32 1

  %vec0 = load volatile <2 x half>, <2 x half> addrspace(3)* %lds, align 4
  %vec1 = load volatile <2 x half>, <2 x half> addrspace(3)* %lds.gep1, align 4
  %scalar0 = load volatile half, half addrspace(3)* %arg2, align 2

  %neg.scalar0 = fsub half -0.0, %scalar0
  %neg.scalar0.vec = insertelement <2 x half> undef, half %scalar0, i32 0
  %scalar0.neg.scalar0 = insertelement <2 x half> %neg.scalar0.vec, half %neg.scalar0, i32 1
  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %scalar0.neg.scalar0)
  store <2 x half> %result, <2 x half> addrspace(1)* %out, align 4
  ret void
}

; Apply fneg before broadcast with bitcast
; GCN-LABEL: {{^}}add_vector_neg_bitcast_scalar_lo:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_xor_b32_e32 [[NEG_SCALAR0:v[0-9]+]], 0x8000, [[SCALAR0]]
; GCN-NEXT: v_pk_add_u16 v{{[0-9]+}}, [[VEC0]], [[NEG_SCALAR0]] op_sel_hi:[1,0]{{$}}
define amdgpu_kernel void @add_vector_neg_bitcast_scalar_lo(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(3)* %lds, half addrspace(3)* %arg2) #0 {
bb:
  %vec0 = load volatile <2 x i16>, <2 x i16> addrspace(3)* %lds, align 4
  %scalar0 = load volatile half, half addrspace(3)* %arg2, align 2
  %neg.scalar0 = fsub half -0.0, %scalar0
  %neg.scalar0.bc = bitcast half %neg.scalar0 to i16

  %neg.scalar0.vec = insertelement <2 x i16> undef, i16 %neg.scalar0.bc, i32 0
  %neg.scalar0.broadcast = shufflevector <2 x i16> %neg.scalar0.vec, <2 x i16> undef, <2 x i32> zeroinitializer

  %result = add <2 x i16> %vec0, %neg.scalar0.broadcast
  store <2 x i16> %result, <2 x i16> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fma_vector_vector_scalar_lo_neg_scalar_hi:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR1:v[0-9]+]]

; FIXME: Remove and
; GCN: v_and_b32_e32 [[SCALAR0]], 0xffff, [[SCALAR0]]
; GCN: v_xor_b32_e32 [[SCALAR1]], 0x8000, [[SCALAR1]]
; GCN: v_lshl_or_b32 [[PACKED:v[0-9]+]], [[SCALAR1]], 16, [[SCALAR0]]

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[PACKED]]{{$}}
define amdgpu_kernel void @fma_vector_vector_scalar_lo_neg_scalar_hi(<2 x half> addrspace(1)* %out, <2 x half> addrspace(3)* %lds, half addrspace(3)* %arg2) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, <2 x half> addrspace(3)* %lds, i32 1
  %arg2.gep = getelementptr inbounds half, half addrspace(3)* %arg2, i32 2

  %vec0 = load volatile <2 x half>, <2 x half> addrspace(3)* %lds, align 4
  %vec1 = load volatile <2 x half>, <2 x half> addrspace(3)* %lds.gep1, align 4

  %scalar0 = load volatile half, half addrspace(3)* %arg2, align 2
  %scalar1 = load volatile half, half addrspace(3)* %arg2.gep, align 2

  %neg.scalar1 = fsub half -0.0, %scalar1
  %vec.ins0 = insertelement <2 x half> undef, half %scalar0, i32 0
  %vec2 = insertelement <2 x half> %vec.ins0, half %neg.scalar1, i32 1
  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %vec2)
  store <2 x half> %result, <2 x half> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fma_vector_vector_neg_scalar_lo_scalar_hi:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR1:v[0-9]+]]

; FIXME: Remove and
; GCN: v_and_b32_e32 [[SCALAR0]], 0xffff, [[SCALAR0]]
; GCN: v_lshl_or_b32 [[PACKED:v[0-9]+]], [[SCALAR1]], 16, [[SCALAR0]]

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[PACKED]] neg_lo:[0,0,1] neg_hi:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_neg_scalar_lo_scalar_hi(<2 x half> addrspace(1)* %out, <2 x half> addrspace(3)* %lds, half addrspace(3)* %arg2) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, <2 x half> addrspace(3)* %lds, i32 1
  %arg2.gep = getelementptr inbounds half, half addrspace(3)* %arg2, i32 2

  %vec0 = load volatile <2 x half>, <2 x half> addrspace(3)* %lds, align 4
  %vec1 = load volatile <2 x half>, <2 x half> addrspace(3)* %lds.gep1, align 4

  %scalar0 = load volatile half, half addrspace(3)* %arg2, align 2
  %scalar1 = load volatile half, half addrspace(3)* %arg2.gep, align 2

  %vec.ins0 = insertelement <2 x half> undef, half %scalar0, i32 0
  %vec2 = insertelement <2 x half> %vec.ins0, half %scalar1, i32 1
  %neg.vec2 = fsub <2 x half> <half -0.0, half -0.0>, %vec2

  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %neg.vec2)
  store <2 x half> %result, <2 x half> addrspace(1)* %out, align 4
  ret void
}

declare <2 x half> @llvm.fma.v2f16(<2 x half>, <2 x half>, <2 x half>) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

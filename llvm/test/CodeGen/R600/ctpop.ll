; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i32 @llvm.ctpop.i32(i32) nounwind readnone
declare <2 x i32> @llvm.ctpop.v2i32(<2 x i32>) nounwind readnone
declare <4 x i32> @llvm.ctpop.v4i32(<4 x i32>) nounwind readnone
declare <8 x i32> @llvm.ctpop.v8i32(<8 x i32>) nounwind readnone
declare <16 x i32> @llvm.ctpop.v16i32(<16 x i32>) nounwind readnone

; FUNC-LABEL: @s_ctpop_i32:
; SI: S_LOAD_DWORD [[SVAL:s[0-9]+]],
; SI: S_BCNT1_I32_B32 [[SRESULT:s[0-9]+]], [[SVAL]]
; SI: V_MOV_B32_e32 [[VRESULT:v[0-9]+]], [[SRESULT]]
; SI: BUFFER_STORE_DWORD [[VRESULT]],
; SI: S_ENDPGM

; EG: BCNT_INT
define void @s_ctpop_i32(i32 addrspace(1)* noalias %out, i32 %val) nounwind {
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  store i32 %ctpop, i32 addrspace(1)* %out, align 4
  ret void
}

; XXX - Why 0 in register?
; FUNC-LABEL: @v_ctpop_i32:
; SI: BUFFER_LOAD_DWORD [[VAL:v[0-9]+]],
; SI: V_MOV_B32_e32 [[VZERO:v[0-9]+]], 0
; SI: V_BCNT_U32_B32_e32 [[RESULT:v[0-9]+]], [[VAL]], [[VZERO]]
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM

; EG: BCNT_INT
define void @v_ctpop_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in) nounwind {
  %val = load i32 addrspace(1)* %in, align 4
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  store i32 %ctpop, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_ctpop_add_chain_i32
; SI: BUFFER_LOAD_DWORD [[VAL0:v[0-9]+]],
; SI: BUFFER_LOAD_DWORD [[VAL1:v[0-9]+]],
; SI: V_MOV_B32_e32 [[VZERO:v[0-9]+]], 0
; SI: V_BCNT_U32_B32_e32 [[MIDRESULT:v[0-9]+]], [[VAL1]], [[VZERO]]
; SI-NOT: ADD
; SI: V_BCNT_U32_B32_e64 [[RESULT:v[0-9]+]], [[VAL0]], [[MIDRESULT]]
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM

; EG: BCNT_INT
; EG: BCNT_INT
define void @v_ctpop_add_chain_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in0, i32 addrspace(1)* noalias %in1) nounwind {
  %val0 = load i32 addrspace(1)* %in0, align 4
  %val1 = load i32 addrspace(1)* %in1, align 4
  %ctpop0 = call i32 @llvm.ctpop.i32(i32 %val0) nounwind readnone
  %ctpop1 = call i32 @llvm.ctpop.i32(i32 %val1) nounwind readnone
  %add = add i32 %ctpop0, %ctpop1
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_ctpop_v2i32:
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: S_ENDPGM

; EG: BCNT_INT
; EG: BCNT_INT
define void @v_ctpop_v2i32(<2 x i32> addrspace(1)* noalias %out, <2 x i32> addrspace(1)* noalias %in) nounwind {
  %val = load <2 x i32> addrspace(1)* %in, align 8
  %ctpop = call <2 x i32> @llvm.ctpop.v2i32(<2 x i32> %val) nounwind readnone
  store <2 x i32> %ctpop, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @v_ctpop_v4i32:
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: S_ENDPGM

; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
define void @v_ctpop_v4i32(<4 x i32> addrspace(1)* noalias %out, <4 x i32> addrspace(1)* noalias %in) nounwind {
  %val = load <4 x i32> addrspace(1)* %in, align 16
  %ctpop = call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %val) nounwind readnone
  store <4 x i32> %ctpop, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: @v_ctpop_v8i32:
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: S_ENDPGM

; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
define void @v_ctpop_v8i32(<8 x i32> addrspace(1)* noalias %out, <8 x i32> addrspace(1)* noalias %in) nounwind {
  %val = load <8 x i32> addrspace(1)* %in, align 32
  %ctpop = call <8 x i32> @llvm.ctpop.v8i32(<8 x i32> %val) nounwind readnone
  store <8 x i32> %ctpop, <8 x i32> addrspace(1)* %out, align 32
  ret void
}

; FUNC-LABEL: @v_ctpop_v16i32:
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: V_BCNT_U32_B32_e32
; SI: S_ENDPGM

; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
define void @v_ctpop_v16i32(<16 x i32> addrspace(1)* noalias %out, <16 x i32> addrspace(1)* noalias %in) nounwind {
  %val = load <16 x i32> addrspace(1)* %in, align 32
  %ctpop = call <16 x i32> @llvm.ctpop.v16i32(<16 x i32> %val) nounwind readnone
  store <16 x i32> %ctpop, <16 x i32> addrspace(1)* %out, align 32
  ret void
}

; FUNC-LABEL: @v_ctpop_i32_add_inline_constant:
; SI: BUFFER_LOAD_DWORD [[VAL:v[0-9]+]],
; SI: V_BCNT_U32_B32_e64 [[RESULT:v[0-9]+]], [[VAL]], 4
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM

; EG: BCNT_INT
define void @v_ctpop_i32_add_inline_constant(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in) nounwind {
  %val = load i32 addrspace(1)* %in, align 4
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  %add = add i32 %ctpop, 4
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_ctpop_i32_add_inline_constant_inv:
; SI: BUFFER_LOAD_DWORD [[VAL:v[0-9]+]],
; SI: V_BCNT_U32_B32_e64 [[RESULT:v[0-9]+]], [[VAL]], 4
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM

; EG: BCNT_INT
define void @v_ctpop_i32_add_inline_constant_inv(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in) nounwind {
  %val = load i32 addrspace(1)* %in, align 4
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  %add = add i32 4, %ctpop
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_ctpop_i32_add_literal:
; SI: BUFFER_LOAD_DWORD [[VAL:v[0-9]+]],
; SI: V_MOV_B32_e32 [[LIT:v[0-9]+]], 0x1869f
; SI: V_BCNT_U32_B32_e32 [[RESULT:v[0-9]+]], [[VAL]], [[LIT]]
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM
define void @v_ctpop_i32_add_literal(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in) nounwind {
  %val = load i32 addrspace(1)* %in, align 4
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  %add = add i32 %ctpop, 99999
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_ctpop_i32_add_var:
; SI-DAG: BUFFER_LOAD_DWORD [[VAL:v[0-9]+]],
; SI-DAG: S_LOAD_DWORD [[VAR:s[0-9]+]],
; SI: V_BCNT_U32_B32_e64 [[RESULT:v[0-9]+]], [[VAL]], [[VAR]]
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM

; EG: BCNT_INT
define void @v_ctpop_i32_add_var(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in, i32 %const) nounwind {
  %val = load i32 addrspace(1)* %in, align 4
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  %add = add i32 %ctpop, %const
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_ctpop_i32_add_var_inv:
; SI-DAG: BUFFER_LOAD_DWORD [[VAL:v[0-9]+]],
; SI-DAG: S_LOAD_DWORD [[VAR:s[0-9]+]],
; SI: V_BCNT_U32_B32_e64 [[RESULT:v[0-9]+]], [[VAL]], [[VAR]]
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM

; EG: BCNT_INT
define void @v_ctpop_i32_add_var_inv(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in, i32 %const) nounwind {
  %val = load i32 addrspace(1)* %in, align 4
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  %add = add i32 %const, %ctpop
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_ctpop_i32_add_vvar_inv
; SI-DAG: BUFFER_LOAD_DWORD [[VAL:v[0-9]+]], {{.*}} + 0x0
; SI-DAG: BUFFER_LOAD_DWORD [[VAR:v[0-9]+]], {{.*}} + 0x10
; SI: V_BCNT_U32_B32_e32 [[RESULT:v[0-9]+]], [[VAL]], [[VAR]]
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM

; EG: BCNT_INT
define void @v_ctpop_i32_add_vvar_inv(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in, i32 addrspace(1)* noalias %constptr) nounwind {
  %val = load i32 addrspace(1)* %in, align 4
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  %gep = getelementptr i32 addrspace(1)* %constptr, i32 4
  %const = load i32 addrspace(1)* %gep, align 4
  %add = add i32 %const, %ctpop
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FIXME: We currently disallow SALU instructions in all branches,
; but there are some cases when the should be allowed.

; FUNC-LABEL: @ctpop_i32_in_br
; SI: BUFFER_LOAD_DWORD [[VAL:v[0-9]+]],
; SI: V_BCNT_U32_B32_e64 [[RESULT:v[0-9]+]], [[VAL]], 0
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM
; EG: BCNT_INT
define void @ctpop_i32_in_br(i32 addrspace(1)* %out, i32 addrspace(1)* %in, i32 %cond) {
entry:
  %0 = icmp eq i32 %cond, 0
  br i1 %0, label %if, label %else

if:
  %1 = load i32 addrspace(1)* %in
  %2 = call i32 @llvm.ctpop.i32(i32 %1)
  br label %endif

else:
  %3 = getelementptr i32 addrspace(1)* %in, i32 1
  %4 = load i32 addrspace(1)* %3
  br label %endif

endif:
  %5 = phi i32 [%2, %if], [%4, %else]
  store i32 %5, i32 addrspace(1)* %out
  ret void
}

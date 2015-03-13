; RUN: llc -march=amdgcn -mcpu=SI < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.r600.read.tidig.x() nounwind readnone

; FUNC-LABEL: @v_test_imax_sge_i32
; SI: v_max_i32_e32
define void @v_test_imax_sge_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %cmp = icmp sge i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @s_test_imax_sge_i32
; SI: s_max_i32
define void @s_test_imax_sge_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp sge i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imax_sge_imm_i32:
; SI: s_max_i32 {{s[0-9]+}}, {{s[0-9]+}}, 9
define void @s_test_imax_sge_imm_i32(i32 addrspace(1)* %out, i32 %a) nounwind {
  %cmp = icmp sge i32 %a, 9
  %val = select i1 %cmp, i32 %a, i32 9
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imax_sgt_imm_i32:
; SI: s_max_i32 {{s[0-9]+}}, {{s[0-9]+}}, 9
define void @s_test_imax_sgt_imm_i32(i32 addrspace(1)* %out, i32 %a) nounwind {
  %cmp = icmp sgt i32 %a, 9
  %val = select i1 %cmp, i32 %a, i32 9
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_imax_sgt_i32
; SI: v_max_i32_e32
define void @v_test_imax_sgt_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %cmp = icmp sgt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @s_test_imax_sgt_i32
; SI: s_max_i32
define void @s_test_imax_sgt_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp sgt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_umax_uge_i32
; SI: v_max_u32_e32
define void @v_test_umax_uge_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %cmp = icmp uge i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @s_test_umax_uge_i32
; SI: s_max_u32
define void @s_test_umax_uge_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp uge i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_umax_ugt_i32
; SI: v_max_u32_e32
define void @v_test_umax_ugt_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %cmp = icmp ugt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @s_test_umax_ugt_i32
; SI: s_max_u32
define void @s_test_umax_ugt_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp ugt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

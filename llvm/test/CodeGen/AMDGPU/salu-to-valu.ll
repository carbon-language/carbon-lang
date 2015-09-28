; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s

declare i32 @llvm.r600.read.tidig.x() #0
declare i32 @llvm.r600.read.tidig.y() #0

; In this test both the pointer and the offset operands to the
; BUFFER_LOAD instructions end up being stored in vgprs.  This
; requires us to add the pointer and offset together, store the
; result in the offset operand (vaddr), and then store 0 in an
; sgpr register pair and use that for the pointer operand
; (low 64-bits of srsrc).

; GCN-LABEL: {{^}}mubuf:

; Make sure we aren't using VGPRs for the source operand of s_mov_b64
; GCN-NOT: s_mov_b64 s[{{[0-9]+:[0-9]+}}], v

; Make sure we aren't using VGPR's for the srsrc operand of BUFFER_LOAD_*
; instructions
; GCN: buffer_load_ubyte v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64
; GCN: buffer_load_ubyte v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64

define void @mubuf(i32 addrspace(1)* %out, i8 addrspace(1)* %in) #1 {
entry:
  %tmp = call i32 @llvm.r600.read.tidig.x()
  %tmp1 = call i32 @llvm.r600.read.tidig.y()
  %tmp2 = sext i32 %tmp to i64
  %tmp3 = sext i32 %tmp1 to i64
  br label %loop

loop:                                             ; preds = %loop, %entry
  %tmp4 = phi i64 [ 0, %entry ], [ %tmp5, %loop ]
  %tmp5 = add i64 %tmp2, %tmp4
  %tmp6 = getelementptr i8, i8 addrspace(1)* %in, i64 %tmp5
  %tmp7 = load i8, i8 addrspace(1)* %tmp6, align 1
  %tmp8 = or i64 %tmp5, 1
  %tmp9 = getelementptr i8, i8 addrspace(1)* %in, i64 %tmp8
  %tmp10 = load i8, i8 addrspace(1)* %tmp9, align 1
  %tmp11 = add i8 %tmp7, %tmp10
  %tmp12 = sext i8 %tmp11 to i32
  store i32 %tmp12, i32 addrspace(1)* %out
  %tmp13 = icmp slt i64 %tmp5, 10
  br i1 %tmp13, label %loop, label %done

done:                                             ; preds = %loop
  ret void
}

; Test moving an SMRD instruction to the VALU

; GCN-LABEL: {{^}}smrd_valu:
; GCN: buffer_load_dword [[OUT:v[0-9]+]]
; GCN: buffer_store_dword [[OUT]]
define void @smrd_valu(i32 addrspace(2)* addrspace(1)* %in, i32 %a, i32 %b, i32 addrspace(1)* %out) #1 {
entry:
  %tmp = icmp ne i32 %a, 0
  br i1 %tmp, label %if, label %else

if:                                               ; preds = %entry
  %tmp1 = load i32 addrspace(2)*, i32 addrspace(2)* addrspace(1)* %in
  br label %endif

else:                                             ; preds = %entry
  %tmp2 = getelementptr i32 addrspace(2)*, i32 addrspace(2)* addrspace(1)* %in
  %tmp3 = load i32 addrspace(2)*, i32 addrspace(2)* addrspace(1)* %tmp2
  br label %endif

endif:                                            ; preds = %else, %if
  %tmp4 = phi i32 addrspace(2)* [ %tmp1, %if ], [ %tmp3, %else ]
  %tmp5 = getelementptr i32, i32 addrspace(2)* %tmp4, i32 3000
  %tmp6 = load i32, i32 addrspace(2)* %tmp5
  store i32 %tmp6, i32 addrspace(1)* %out
  ret void
}

; Test moving an SMRD with an immediate offset to the VALU

; GCN-LABEL: {{^}}smrd_valu2:
; GCN: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
define void @smrd_valu2(i32 addrspace(1)* %out, [8 x i32] addrspace(2)* %in) #1 {
entry:
  %tmp = call i32 @llvm.r600.read.tidig.x() #0
  %tmp1 = add i32 %tmp, 4
  %tmp2 = getelementptr [8 x i32], [8 x i32] addrspace(2)* %in, i32 %tmp, i32 4
  %tmp3 = load i32, i32 addrspace(2)* %tmp2
  store i32 %tmp3, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}smrd_valu2_max_smrd_offset:
; GCN: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:1020{{$}}
define void @smrd_valu2_max_smrd_offset(i32 addrspace(1)* %out, [1024 x i32] addrspace(2)* %in) #1 {
entry:
  %tmp = call i32 @llvm.r600.read.tidig.x() #0
  %tmp1 = add i32 %tmp, 4
  %tmp2 = getelementptr [1024 x i32], [1024 x i32] addrspace(2)* %in, i32 %tmp, i32 255
  %tmp3 = load i32, i32 addrspace(2)* %tmp2
  store i32 %tmp3, i32 addrspace(1)* %out
  ret void
}

; Offset is too big to fit in SMRD 8-bit offset, but small enough to
; fit in MUBUF offset.
; FIXME: We should be using the offset but we don't

; GCN-LABEL: {{^}}smrd_valu2_mubuf_offset:
; GCN: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
define void @smrd_valu2_mubuf_offset(i32 addrspace(1)* %out, [1024 x i32] addrspace(2)* %in) #1 {
entry:
  %tmp = call i32 @llvm.r600.read.tidig.x() #0
  %tmp1 = add i32 %tmp, 4
  %tmp2 = getelementptr [1024 x i32], [1024 x i32] addrspace(2)* %in, i32 %tmp, i32 256
  %tmp3 = load i32, i32 addrspace(2)* %tmp2
  store i32 %tmp3, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_load_imm_v8i32:
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx4
define void @s_load_imm_v8i32(<8 x i32> addrspace(1)* %out, i32 addrspace(2)* nocapture readonly %in) #1 {
entry:
  %tmp0 = tail call i32 @llvm.r600.read.tidig.x()
  %tmp1 = getelementptr inbounds i32, i32 addrspace(2)* %in, i32 %tmp0
  %tmp2 = bitcast i32 addrspace(2)* %tmp1 to <8 x i32> addrspace(2)*
  %tmp3 = load <8 x i32>, <8 x i32> addrspace(2)* %tmp2, align 4
  store <8 x i32> %tmp3, <8 x i32> addrspace(1)* %out, align 32
  ret void
}

; GCN-LABEL: {{^}}s_load_imm_v16i32:
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx4
define void @s_load_imm_v16i32(<16 x i32> addrspace(1)* %out, i32 addrspace(2)* nocapture readonly %in) #1 {
entry:
  %tmp0 = tail call i32 @llvm.r600.read.tidig.x() #1
  %tmp1 = getelementptr inbounds i32, i32 addrspace(2)* %in, i32 %tmp0
  %tmp2 = bitcast i32 addrspace(2)* %tmp1 to <16 x i32> addrspace(2)*
  %tmp3 = load <16 x i32>, <16 x i32> addrspace(2)* %tmp2, align 4
  store <16 x i32> %tmp3, <16 x i32> addrspace(1)* %out, align 32
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

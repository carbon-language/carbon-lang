; RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs | FileCheck %s

; In this test both the pointer and the offset operands to the
; BUFFER_LOAD instructions end up being stored in vgprs.  This
; requires us to add the pointer and offset together, store the
; result in the offset operand (vaddr), and then store 0 in an
; sgpr register pair and use that for the pointer operand
; (low 64-bits of srsrc).

; CHECK-LABEL: {{^}}mubuf:

; Make sure we aren't using VGPRs for the source operand of s_mov_b64
; CHECK-NOT: s_mov_b64 s[{{[0-9]+:[0-9]+}}], v

; Make sure we aren't using VGPR's for the srsrc operand of BUFFER_LOAD_*
; instructions
; CHECK: buffer_load_ubyte v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64
; CHECK: buffer_load_ubyte v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64
define void @mubuf(i32 addrspace(1)* %out, i8 addrspace(1)* %in) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.x() #1
  %1 = call i32 @llvm.r600.read.tidig.y() #1
  %2 = sext i32 %0 to i64
  %3 = sext i32 %1 to i64
  br label %loop

loop:
  %4 = phi i64 [0, %entry], [%5, %loop]
  %5 = add i64 %2, %4
  %6 = getelementptr i8, i8 addrspace(1)* %in, i64 %5
  %7 = load i8, i8 addrspace(1)* %6, align 1
  %8 = or i64 %5, 1
  %9 = getelementptr i8, i8 addrspace(1)* %in, i64 %8
  %10 = load i8, i8 addrspace(1)* %9, align 1
  %11 = add i8 %7, %10
  %12 = sext i8 %11 to i32
  store i32 %12, i32 addrspace(1)* %out
  %13 = icmp slt i64 %5, 10
  br i1 %13, label %loop, label %done

done:
  ret void
}

declare i32 @llvm.r600.read.tidig.x() #1
declare i32 @llvm.r600.read.tidig.y() #1

attributes #1 = { nounwind readnone }

; Test moving an SMRD instruction to the VALU

; CHECK-LABEL: {{^}}smrd_valu:
; CHECK: buffer_load_dword [[OUT:v[0-9]+]]
; CHECK: buffer_store_dword [[OUT]]

define void @smrd_valu(i32 addrspace(2)* addrspace(1)* %in, i32 %a, i32 addrspace(1)* %out) {
entry:
  %0 = icmp ne i32 %a, 0
  br i1 %0, label %if, label %else

if:
  %1 = load i32 addrspace(2)*, i32 addrspace(2)* addrspace(1)* %in
  br label %endif

else:
  %2 = getelementptr i32 addrspace(2)*, i32 addrspace(2)* addrspace(1)* %in
  %3 = load i32 addrspace(2)*, i32 addrspace(2)* addrspace(1)* %2
  br label %endif

endif:
  %4 = phi i32 addrspace(2)*  [%1, %if], [%3, %else]
  %5 = getelementptr i32, i32 addrspace(2)* %4, i32 3000
  %6 = load i32, i32 addrspace(2)* %5
  store i32 %6, i32 addrspace(1)* %out
  ret void
}

; Test moving an SMRD with an immediate offset to the VALU

; CHECK-LABEL: {{^}}smrd_valu2:
; CHECK: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
define void @smrd_valu2(i32 addrspace(1)* %out, [8 x i32] addrspace(2)* %in) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %1 = add i32 %0, 4
  %2 = getelementptr [8 x i32], [8 x i32] addrspace(2)* %in, i32 %0, i32 4
  %3 = load i32, i32 addrspace(2)* %2
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}smrd_valu2_max_smrd_offset:
; CHECK: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:1020{{$}}
define void @smrd_valu2_max_smrd_offset(i32 addrspace(1)* %out, [1024 x i32] addrspace(2)* %in) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %1 = add i32 %0, 4
  %2 = getelementptr [1024 x i32], [1024 x i32] addrspace(2)* %in, i32 %0, i32 255
  %3 = load i32, i32 addrspace(2)* %2
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; Offset is too big to fit in SMRD 8-bit offset, but small enough to
; fit in MUBUF offset.
; FIXME: We should be using the offset but we don't

; CHECK-LABEL: {{^}}smrd_valu2_mubuf_offset:
; CHECK: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
define void @smrd_valu2_mubuf_offset(i32 addrspace(1)* %out, [1024 x i32] addrspace(2)* %in) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %1 = add i32 %0, 4
  %2 = getelementptr [1024 x i32], [1024 x i32] addrspace(2)* %in, i32 %0, i32 256
  %3 = load i32, i32 addrspace(2)* %2
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}s_load_imm_v8i32:
; CHECK: buffer_load_dwordx4
; CHECK: buffer_load_dwordx4
define void @s_load_imm_v8i32(<8 x i32> addrspace(1)* %out, i32 addrspace(2)* nocapture readonly %in) {
entry:
  %tmp0 = tail call i32 @llvm.r600.read.tidig.x() #1
  %tmp1 = getelementptr inbounds i32, i32 addrspace(2)* %in, i32 %tmp0
  %tmp2 = bitcast i32 addrspace(2)* %tmp1 to <8 x i32> addrspace(2)*
  %tmp3 = load <8 x i32>, <8 x i32> addrspace(2)* %tmp2, align 4
  store <8 x i32> %tmp3, <8 x i32> addrspace(1)* %out, align 32
  ret void
}

; CHECK-LABEL: {{^}}s_load_imm_v16i32:
; CHECK: buffer_load_dwordx4
; CHECK: buffer_load_dwordx4
; CHECK: buffer_load_dwordx4
; CHECK: buffer_load_dwordx4
define void @s_load_imm_v16i32(<16 x i32> addrspace(1)* %out, i32 addrspace(2)* nocapture readonly %in) {
entry:
  %tmp0 = tail call i32 @llvm.r600.read.tidig.x() #1
  %tmp1 = getelementptr inbounds i32, i32 addrspace(2)* %in, i32 %tmp0
  %tmp2 = bitcast i32 addrspace(2)* %tmp1 to <16 x i32> addrspace(2)*
  %tmp3 = load <16 x i32>, <16 x i32> addrspace(2)* %tmp2, align 4
  store <16 x i32> %tmp3, <16 x i32> addrspace(1)* %out, align 32
  ret void
}

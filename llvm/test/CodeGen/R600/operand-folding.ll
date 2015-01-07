; RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: {{^}}fold_sgpr:
; CHECK: v_add_i32_e32 v{{[0-9]+}}, s
define void @fold_sgpr(i32 addrspace(1)* %out, i32 %fold) {
entry:
  %tmp0 = icmp ne i32 %fold, 0
  br i1 %tmp0, label %if, label %endif

if:
  %id = call i32 @llvm.r600.read.tidig.x()
  %offset = add i32 %fold, %id
  %tmp1 = getelementptr i32 addrspace(1)* %out, i32 %offset
  store i32 0, i32 addrspace(1)* %tmp1
  br label %endif

endif:
  ret void
}

; CHECK-LABEL: {{^}}fold_imm:
; CHECK v_or_i32_e32 v{{[0-9]+}}, 5
define void @fold_imm(i32 addrspace(1)* %out, i32 %cmp) {
entry:
  %fold = add i32 3, 2
  %tmp0 = icmp ne i32 %cmp, 0
  br i1 %tmp0, label %if, label %endif

if:
  %id = call i32 @llvm.r600.read.tidig.x()
  %val = or i32 %id, %fold
  store i32 %val, i32 addrspace(1)* %out
  br label %endif

endif:
  ret void
}

; CHECK-LABEL: {{^}}fold_64bit_constant_add:
; CHECK-NOT: s_mov_b64
; FIXME: It would be better if we could use v_add here and drop the extra
; v_mov_b32 instructions.
; CHECK-DAG: s_add_u32 [[LO:s[0-9]+]], s{{[0-9]+}}, 1
; CHECK-DAG: s_addc_u32 [[HI:s[0-9]+]], s{{[0-9]+}}, 0
; CHECK-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], [[LO]]
; CHECK-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], [[HI]]
; CHECK: buffer_store_dwordx2 v{{\[}}[[VLO]]:[[VHI]]{{\]}},

define void @fold_64bit_constant_add(i64 addrspace(1)* %out, i32 %cmp, i64 %val) {
entry:
  %tmp0 = add i64 %val, 1
  store i64 %tmp0, i64 addrspace(1)* %out
  ret void
}

declare i32 @llvm.r600.read.tidig.x() #0
attributes #0 = { readnone }

; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s

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

declare i32 @llvm.r600.read.tidig.x() #0
attributes #0 = { readnone }

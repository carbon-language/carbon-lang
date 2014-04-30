; RUN: llc < %s -march=r600 -mcpu=SI | FileCheck --check-prefix=SI %s

; Make sure the i1 values created by the cfg structurizer pass are
; moved using VALU instructions
; SI-NOT: S_MOV_B64 s[{{[0-9]:[0-9]}}], -1
; SI: V_MOV_B32_e32 v{{[0-9]}}, -1
define void @test_if(i32 %a, i32 %b, i32 addrspace(1)* %src, i32 addrspace(1)* %dst) {
entry:
  switch i32 %a, label %default [
    i32 0, label %case0
    i32 1, label %case1
  ]

case0:
  %arrayidx1 = getelementptr i32 addrspace(1)* %dst, i32 %b
  store i32 0, i32 addrspace(1)* %arrayidx1, align 4
  br label %end

case1:
  %arrayidx5 = getelementptr i32 addrspace(1)* %dst, i32 %b
  store i32 1, i32 addrspace(1)* %arrayidx5, align 4
  br label %end

default:
  %cmp8 = icmp eq i32 %a, 2
  %arrayidx10 = getelementptr i32 addrspace(1)* %dst, i32 %b
  br i1 %cmp8, label %if, label %else

if:
  store i32 2, i32 addrspace(1)* %arrayidx10, align 4
  br label %end

else:
  store i32 3, i32 addrspace(1)* %arrayidx10, align 4
  br label %end

end:
  ret void
}

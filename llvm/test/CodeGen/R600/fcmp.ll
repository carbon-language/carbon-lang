;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK: SETE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;CHECK: MOV T{{[0-9]+\.[XYZW], -T[0-9]+\.[XYZW]}}
;CHECK: FLT_TO_INT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @test(i32 addrspace(1)* %out, float addrspace(1)* %in) {
entry:
  %0 = load float addrspace(1)* %in
  %arrayidx1 = getelementptr inbounds float addrspace(1)* %in, i32 1
  %1 = load float addrspace(1)* %arrayidx1
  %cmp = fcmp oeq float %0, %1
  %sext = sext i1 %cmp to i32
  store i32 %sext, i32 addrspace(1)* %out
  ret void
}

;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK-NOT: SETE
;CHECK: CNDE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, 1.0, literal.x,
;CHECK-NEXT: {{[-0-9]+\(2.0}}
define void @test(float addrspace(1)* %out, float addrspace(1)* %in) {
  %1 = load float addrspace(1)* %in
  %2 = fcmp oeq float %1, 0.0
  %3 = select i1 %2, float 1.0, float 2.0
  store float %3, float addrspace(1)* %out
  ret void
}

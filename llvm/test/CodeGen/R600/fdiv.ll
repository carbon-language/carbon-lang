;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}
;CHECK-DAG: MUL_IEEE  T{{[0-9]+\.[XYZW]}}
;CHECK-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}
;CHECK-DAG: MUL_IEEE  T{{[0-9]+\.[XYZW]}}
;CHECK-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}
;CHECK-DAG: MUL_IEEE  T{{[0-9]+\.[XYZW]}}
;CHECK-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}
;CHECK-DAG: MUL_IEEE  T{{[0-9]+\.[XYZW]}}

define void @test(<4 x float> addrspace(1)* %out, <4 x float> %a, <4 x float> %b) {
entry:
  %0 = fdiv <4 x float> %a, %b
  store <4 x float> %0, <4 x float> addrspace(1)* %out
  ret void
}

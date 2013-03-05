;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK: RECIP_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;CHECK: MUL_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;CHECK: RECIP_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;CHECK: MUL_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;CHECK: RECIP_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;CHECK: MUL_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;CHECK: RECIP_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;CHECK: MUL_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @test(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x float> addrspace(1)* %in, i32 1
  %a = load <4 x float> addrspace(1) * %in
  %b = load <4 x float> addrspace(1) * %b_ptr
  %result = fdiv <4 x float> %a, %b
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}

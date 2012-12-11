;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK: RAT_WRITE_CACHELESS_128 T{{[0-9]+\.XYZW, T[0-9]+\.X}}, 1

define void @test(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
  %1 = load <4 x float> addrspace(1) * %in
  store <4 x float> %1, <4 x float> addrspace(1)* %out
  ret void
}

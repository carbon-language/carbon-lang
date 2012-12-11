;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK: VTX_READ_32 T{{[0-9]+\.X, T[0-9]+\.X}}

define void @test(float addrspace(1)* %out, float addrspace(2)* %in) {
  %1 = load float addrspace(2)* %in
  store float %1, float addrspace(1)* %out
  ret void
}

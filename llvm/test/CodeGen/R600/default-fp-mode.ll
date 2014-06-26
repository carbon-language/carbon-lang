; RUN: llc -march=r600 -mcpu=SI < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: @test_kernel
; SI: FloatMode: 240
; SI: IeeeMode: 0
define void @test_kernel(float addrspace(1)* %out0, double addrspace(1)* %out1) nounwind {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

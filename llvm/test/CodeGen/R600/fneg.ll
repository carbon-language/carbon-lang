; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; XXX: There is a bug in the DAGCombiner that lowers fneg to XOR, this test
; will need to be changed once it is fixed.

; CHECK: @fneg_v2
; CHECK: XOR_INT
; CHECK: XOR_INT
define void @fneg_v2(<2 x float> addrspace(1)* nocapture %out, <2 x float> %in) {
entry:
  %0 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %in
  store <2 x float> %0, <2 x float> addrspace(1)* %out
  ret void
}

; CHECK: @fneg_v4
; CHECK: XOR_INT
; CHECK: XOR_INT
; CHECK: XOR_INT
; CHECK: XOR_INT
define void @fneg_v4(<4 x float> addrspace(1)* nocapture %out, <4 x float> %in) {
entry:
  %0 = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %in
  store <4 x float> %0, <4 x float> addrspace(1)* %out
  ret void
}

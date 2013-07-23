; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: @fneg_v2
; CHECK: -PV
; CHECK: -PV
define void @fneg_v2(<2 x float> addrspace(1)* nocapture %out, <2 x float> %in) {
entry:
  %0 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %in
  store <2 x float> %0, <2 x float> addrspace(1)* %out
  ret void
}

; CHECK: @fneg_v4
; CHECK: -PV
; CHECK: -PV
; CHECK: -PV
; CHECK: -PV
define void @fneg_v4(<4 x float> addrspace(1)* nocapture %out, <4 x float> %in) {
entry:
  %0 = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %in
  store <4 x float> %0, <4 x float> addrspace(1)* %out
  ret void
}

; DAGCombiner will transform:
; (fneg (f32 bitcast (i32 a))) => (f32 bitcast (xor (i32 a), 0x80000000))
; unless the target returns true for isNegFree()

; CHECK-NOT: XOR
; CHECK: -KC0[2].Z

define void @fneg_free(float addrspace(1)* %out, i32 %in) {
entry:
  %0 = bitcast i32 %in to float
  %1 = fsub float 0.0, %0
  store float %1, float addrspace(1)* %out
  ret void
}

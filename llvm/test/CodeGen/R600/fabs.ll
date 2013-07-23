; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=R600-CHECK
; RUN: llc < %s -march=r600 -mcpu=SI | FileCheck %s --check-prefix=SI-CHECK

; DAGCombiner will transform:
; (fabs (f32 bitcast (i32 a))) => (f32 bitcast (and (i32 a), 0x7FFFFFFF))
; unless isFabsFree returns true

; R600-CHECK: @fabs_free
; R600-CHECK-NOT: AND
; R600-CHECK: |PV.{{[XYZW]}}|
; SI-CHECK: @fabs_free
; SI-CHECK: V_ADD_F32_e64 VGPR{{[0-9]}}, SGPR{{[0-9]}}, 0, 1, 0, 0, 0

define void @fabs_free(float addrspace(1)* %out, i32 %in) {
entry:
  %0 = bitcast i32 %in to float
  %1 = call float @fabs(float %0)
  store float %1, float addrspace(1)* %out
  ret void
}

declare float @fabs(float ) readnone

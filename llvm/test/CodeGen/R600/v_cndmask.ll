; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck --check-prefix=SI %s

; SI: @v_cnd_nan
; SI: V_CNDMASK_B32_e64 v{{[0-9]}},
; SI-DAG: v{{[0-9]}}
; SI-DAG: -nan
define void @v_cnd_nan(float addrspace(1)* %out, i32 %c, float %f) {
entry:
  %0 = icmp ne i32 %c, 0
  %1 = select i1 %0, float 0xFFFFFFFFE0000000, float %f
  store float %1, float addrspace(1)* %out
  ret void
}

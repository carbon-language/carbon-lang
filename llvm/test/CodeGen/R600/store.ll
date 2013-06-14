; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck --check-prefix=CM-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=verde | FileCheck --check-prefix=SI-CHECK %s

; floating-point store
; EG-CHECK: @store_f32
; EG-CHECK: RAT_WRITE_CACHELESS_32_eg T{{[0-9]+\.X, T[0-9]+\.X}}, 1
; CM-CHECK: @store_f32
; CM-CHECK: EXPORT_RAT_INST_STORE_DWORD T{{[0-9]+\.X, T[0-9]+\.X}}
; SI-CHECK: @store_f32
; SI-CHECK: BUFFER_STORE_DWORD

define void @store_f32(float addrspace(1)* %out, float %in) {
  store float %in, float addrspace(1)* %out
  ret void
}

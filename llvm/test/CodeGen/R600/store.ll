; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=verde | FileCheck --check-prefix=SI-CHECK %s

; CHECK: @store_float
; EG-CHECK: RAT_WRITE_CACHELESS_32_eg T{{[0-9]+\.X, T[0-9]+\.X}}, 1
; SI-CHECK: BUFFER_STORE_DWORD

define void @store_float(float addrspace(1)* %out, float %in) {
  store float %in, float addrspace(1)* %out
  ret void
}

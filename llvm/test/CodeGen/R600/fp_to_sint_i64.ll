; FIXME: Merge into fp_to_sint.ll when EG/NI supports 64-bit types
; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck --check-prefix=SI %s

; SI-LABEL: @fp_to_sint_i64
; Check that the compiler doesn't crash with a "cannot select" error
; SI: S_ENDPGM
define void @fp_to_sint_i64 (i64 addrspace(1)* %out, float %in) {
entry:
  %0 = fptosi float %in to i64
  store i64 %0, i64 addrspace(1)* %out
  ret void
}

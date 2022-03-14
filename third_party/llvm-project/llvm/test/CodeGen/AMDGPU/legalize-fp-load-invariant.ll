; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs -stop-after=amdgpu-isel -o - %s | FileCheck -check-prefix=GCN %s

; Type legalization for illegal FP type results was dropping invariant
; and dereferenceable flags.

; GCN: BUFFER_LOAD_USHORT{{.*}} :: (dereferenceable invariant load (s16) from %ir.ptr, addrspace 4)
define half @legalize_f16_load(half addrspace(4)* dereferenceable(4) %ptr) {
  %load = load half, half addrspace(4)* %ptr, !invariant.load !0
  %add = fadd half %load, 1.0
  ret half %add
}

!0 = !{}

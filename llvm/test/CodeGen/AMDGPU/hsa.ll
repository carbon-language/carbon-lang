; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | FileCheck --check-prefix=HSA %s

; HSA: {{^}}simple:
; HSA: .section        .hsa.version
; HSA-NEXT: .ascii  "HSA Code Unit:0.0:AMD:0.1:GFX8.1:0"
; Test that the amd_kernel_code_t object is emitted
; HSA: .asciz
; HSA: s_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[0:1], 0x0
; Make sure we are setting the ATC bit:
; HSA: s_mov_b32 s[[HI:[0-9]]], 0x100f000
; HSA: buffer_store_dword v{{[0-9]+}}, s[0:[[HI]]], 0

define void @simple(i32 addrspace(1)* %out) {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}

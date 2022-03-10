; RUN: llc -march=amdgcn -mcpu=gfx600 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=NOT-SUPPORTED %s
; RUN: llc -march=amdgcn -mcpu=gfx700 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=NOT-SUPPORTED %s
; RUN: llc -march=amdgcn -mcpu=gfx801 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ANY %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ANY %s
; RUN: llc -march=amdgcn -mcpu=gfx902 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ANY %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ANY %s

; REQUIRES: asserts

; NOT-SUPPORTED: xnack setting for subtarget: Unsupported
; ANY: xnack setting for subtarget: Any
define void @xnack-subtarget-feature-any() #0 {
  ret void
}

attributes #0 = { nounwind }

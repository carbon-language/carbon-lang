; RUN: llc -march=amdgcn -mcpu=gfx700 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=WARN %s
; RUN: llc -march=amdgcn -mcpu=gfx906 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=OFF %s
; RUN: llc -march=amdgcn -mcpu=gfx908 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=OFF %s

; REQUIRES: asserts

; WARN: warning: sramecc 'Off' was requested for a processor that does not support it!
; OFF: sramecc setting for subtarget: Off

define void @sramecc-subtarget-feature-disabled() #0 {
  ret void
}

attributes #0 = { "target-features"="-sramecc" }

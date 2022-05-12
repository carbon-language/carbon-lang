; RUN: llc -march=amdgcn -mcpu=gfx600 -debug-only=amdgpu-subtarget -o /dev/null %s 2>&1 | FileCheck --check-prefix=WARN %s
; RUN: llc -march=amdgcn -mcpu=gfx700 -debug-only=amdgpu-subtarget -o /dev/null %s 2>&1 | FileCheck --check-prefix=WARN %s
; RUN: llc -march=amdgcn -mcpu=gfx801 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -march=amdgcn -mcpu=gfx906 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s

; REQUIRES: asserts

; WARN: warning: xnack 'On' was requested for a processor that does not support it!
; ON: xnack setting for subtarget: On
define void @xnack-subtarget-feature-enabled() #0 {
  ret void
}

attributes #0 = { "target-features"="+xnack" }

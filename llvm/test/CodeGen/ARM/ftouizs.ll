; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,+neonfp | FileCheck %s -check-prefix=NEON
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,-neonfp | FileCheck %s -check-prefix=VFP2
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a8 | FileCheck %s -check-prefix=NEON
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a9 | FileCheck %s -check-prefix=VFP2

define i32 @test(float %a, float %b) {
; VFP2: ftouizs s0, s0
; NEON: vcvt.u32.f32 d0, d0
entry:
        %0 = fadd float %a, %b
        %1 = fptoui float %0 to i32
	ret i32 %1
}

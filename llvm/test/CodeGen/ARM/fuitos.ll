; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,+neonfp | FileCheck %s -check-prefix=NEON
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,-neonfp | FileCheck %s -check-prefix=VFP2
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a8 | FileCheck %s -check-prefix=NEON
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a9 | FileCheck %s -check-prefix=VFP2

define float @test(i32 %a, i32 %b) {
; VFP2: fuitos s0, s0
; NEON: vcvt.f32.u32 d0, d0
entry:
        %0 = add i32 %a, %b
        %1 = uitofp i32 %0 to float
	ret float %1
}

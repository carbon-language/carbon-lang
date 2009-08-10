; XFAIL: *
; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 | FileCheck %s
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,+neonfp | FileCheck %s
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,-neonfp | FileCheck %s
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a8 | FileCheck %s
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a9 | FileCheck %s

define float @test1(float %a, float %b) nounwind {
; CHECK: fnmscs s2, s1, s0 
entry:
	%0 = fmul float %a, %b
        %1 = fsub float -0.0, %0
	ret float %1
}

define float @test2(float %a, float %b) nounwind {
; CHECK: fnmscs s2, s1, s0 
entry:
	%0 = fmul float %a, %b
        %1 = fmul float -1.0, %0
	ret float %1
}


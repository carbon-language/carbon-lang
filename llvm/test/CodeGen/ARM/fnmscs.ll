; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s
; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s
; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s
; RUN: llc < %s -march=arm -mcpu=cortex-a9 | FileCheck %s

define float @test1(float %acc, float %a, float %b) nounwind {
; CHECK: vnmla.f32 s{{.*}}, s{{.*}}, s{{.*}}
entry:
	%0 = fmul float %a, %b
	%1 = fsub float -0.0, %0
        %2 = fsub float %1, %acc
	ret float %2
}

define float @test2(float %acc, float %a, float %b) nounwind {
; CHECK: vnmla.f32 s{{.*}}, s{{.*}}, s{{.*}}
entry:
	%0 = fmul float %a, %b
	%1 = fmul float -1.0, %0
        %2 = fsub float %1, %acc
	ret float %2
}


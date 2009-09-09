; RUN: llc < %s -march=arm -mattr=+vfp2 | grep -E {fabss\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llc < %s -march=arm -mattr=+neon,+neonfp | grep -E {vabs.f32\\W*d\[0-9\]+,\\W*d\[0-9\]+} | count 1
; RUN: llc < %s -march=arm -mattr=+neon,-neonfp | grep -E {fabss\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llc < %s -march=arm -mcpu=cortex-a8 | grep -E {vabs.f32\\W*d\[0-9\]+,\\W*d\[0-9\]+} | count 1
; RUN: llc < %s -march=arm -mcpu=cortex-a9 | grep -E {fabss\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1

define float @test(float %a, float %b) {
entry:
        %dum = fadd float %a, %b
	%0 = tail call float @fabsf(float %dum)
        %dum1 = fadd float %0, %b
	ret float %dum1
}

declare float @fabsf(float)

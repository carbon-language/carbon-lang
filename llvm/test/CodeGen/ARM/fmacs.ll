; RUN: llc < %s -march=arm -mattr=+vfp2 | grep -E {fmacs\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llc < %s -march=arm -mattr=+neon -arm-use-neon-fp=1 | grep -E {vmla.f32\\W*d\[0-9\]+,\\W*d\[0-9\]+,\\W*d\[0-9\]+} | count 1
; RUN: llc < %s -march=arm -mattr=+neon -arm-use-neon-fp=0 | grep -E {fmacs\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llc < %s -march=arm -mcpu=cortex-a8 | grep -E {vmla.f32\\W*d\[0-9\]+,\\W*d\[0-9\]+,\\W*d\[0-9\]+} | count 1
; RUN: llc < %s -march=arm -mcpu=cortex-a9 | grep -E {fmacs\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1

define float @test(float %acc, float %a, float %b) {
entry:
	%0 = fmul float %a, %b
        %1 = fadd float %acc, %0
	ret float %1
}


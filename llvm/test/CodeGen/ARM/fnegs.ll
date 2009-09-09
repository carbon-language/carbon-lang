; RUN: llc < %s -march=arm -mattr=+vfp2 | grep -E {fnegs\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 2
; RUN: llc < %s -march=arm -mattr=+neon,+neonfp | grep -E {vneg.f32\\W*d\[0-9\]+,\\W*d\[0-9\]+} | count 2
; RUN: llc < %s -march=arm -mattr=+neon,-neonfp | grep -E {fnegs\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 2
; RUN: llc < %s -march=arm -mcpu=cortex-a8 | grep -E {vneg.f32\\W*d\[0-9\]+,\\W*d\[0-9\]+} | count 2
; RUN: llc < %s -march=arm -mcpu=cortex-a9 | grep -E {fnegs\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 2

define float @test1(float* %a) {
entry:
	%0 = load float* %a, align 4		; <float> [#uses=2]
	%1 = fsub float -0.000000e+00, %0		; <float> [#uses=2]
	%2 = fpext float %1 to double		; <double> [#uses=1]
	%3 = fcmp olt double %2, 1.234000e+00		; <i1> [#uses=1]
	%retval = select i1 %3, float %1, float %0		; <float> [#uses=1]
	ret float %retval
}

define float @test2(float* %a) {
entry:
	%0 = load float* %a, align 4		; <float> [#uses=2]
	%1 = fmul float -1.000000e+00, %0		; <float> [#uses=2]
	%2 = fpext float %1 to double		; <double> [#uses=1]
	%3 = fcmp olt double %2, 1.234000e+00		; <i1> [#uses=1]
	%retval = select i1 %3, float %1, float %0		; <float> [#uses=1]
	ret float %retval
}

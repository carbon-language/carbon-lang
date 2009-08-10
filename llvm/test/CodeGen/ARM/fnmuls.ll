; XFAIL: *
; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 | grep -E {fnmuls\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 2
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,+neonfp | grep -E {fnmuls\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 2
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,-neonfp | grep -E {fnmuls\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 2
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a8 | grep -E {fnmuls\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 2
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a9 | grep -E {fnmuls\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 2

define float @test1(float %a, float %b) nounwind {
entry:
	%0 = fmul float %a, %b
        %1 = fsub float -0.0, %0
	ret float %1
}

define float @test2(float %a, float %b) nounwind {
entry:
	%0 = fmul float %a, %b
        %1 = fmul float -1.0, %0
	ret float %1
}


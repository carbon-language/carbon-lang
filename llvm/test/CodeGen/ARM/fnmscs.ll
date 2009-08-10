; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 | grep -E {fnmscs\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 2
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,+neonfp | grep -E {fnmscs\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 2
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,-neonfp | grep -E {fnmscs\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 2
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a8 | grep -E {fnmscs\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 2
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a9 | grep -E {fnmscs\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 2

define float @test1(float %acc, float %a, float %b) nounwind {
entry:
	%0 = fmul float %a, %b
	%1 = fsub float -0.0, %0
        %2 = fsub float %1, %acc
	ret float %2
}

define float @test2(float %acc, float %a, float %b) nounwind {
entry:
	%0 = fmul float %a, %b
	%1 = fmul float -1.0, %0
        %2 = fsub float %1, %acc
	ret float %2
}


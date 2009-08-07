; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 | grep -E {fdivs\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,+neonfp | grep -E {fdivs\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,-neonfp | grep -E {fdivs\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a8 | grep -E {fdivs\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a9 | grep -E {fdivs\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1

define float @test(float %a, float %b) {
entry:
	%0 = fdiv float %a, %b
	ret float %0
}


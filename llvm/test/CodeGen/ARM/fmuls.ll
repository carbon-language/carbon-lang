; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 | grep -E {fmuls\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,+neonfp | grep -E {vmul.f32\\W*d\[0-9\]+,\\W*d\[0-9\]+,\\W*d\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,-neonfp | grep -E {fmuls\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a8 | grep -E {vmul.f32\\W*d\[0-9\]+,\\W*d\[0-9\]+,\\W*d\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a9 | grep -E {fmuls\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1

define float @test(float %a, float %b) {
entry:
	%0 = fmul float %a, %b
	ret float %0
}


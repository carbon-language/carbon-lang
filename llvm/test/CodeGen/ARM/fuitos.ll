; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 | grep -E {fuitos\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,+neonfp | grep -E {vcvt.f32.u32\\W*d\[0-9\]+,\\W*d\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,-neonfp | grep -E {fuitos\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a8 | grep -E {vcvt.f32.u32\\W*d\[0-9\]+,\\W*d\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a9 | grep -E {fuitos\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1

define float @test(i32 %a, i32 %b) {
entry:
        %0 = add i32 %a, %b
        %1 = uitofp i32 %0 to float
	ret float %1
}

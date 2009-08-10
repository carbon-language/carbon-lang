; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 | grep -E {ftosizs\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,+neonfp | grep -E {vcvt.s32.f32\\W*d\[0-9\]+,\\W*d\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,-neonfp | grep -E {ftosizs\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a8 | grep -E {vcvt.s32.f32\\W*d\[0-9\]+,\\W*d\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a9 | grep -E {ftosizs\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1

define i32 @test(float %a, float %b) {
entry:
        %0 = fadd float %a, %b
        %1 = fptosi float %0 to i32
	ret i32 %1
}

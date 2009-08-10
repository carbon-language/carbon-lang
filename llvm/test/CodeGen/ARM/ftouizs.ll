; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 | grep -E {ftouizs\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,+neonfp | grep -E {vcvt.u32.f32\\W*d\[0-9\]+,\\W*d\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,-neonfp | grep -E {ftouizs\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a8 | grep -E {vcvt.u32.f32\\W*d\[0-9\]+,\\W*d\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mcpu=cortex-a9 | grep -E {ftouizs\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1

define i32 @test(float %a, float %b) {
entry:
        %0 = fadd float %a, %b
        %1 = fptoui float %0 to i32
	ret i32 %1
}

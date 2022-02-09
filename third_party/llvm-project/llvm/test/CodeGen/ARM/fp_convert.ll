; RUN: llc -mtriple=arm-eabi -mattr=+vfp2 %s -o - \
; RUN:  | FileCheck %s -check-prefix=VFP2

; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - \
; RUN:  | FileCheck %s -check-prefix=VFP2

; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 %s -o - \
; RUN: | FileCheck %s -check-prefix=VFP2

; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 --enable-unsafe-fp-math %s -o - \
; RUN:  | FileCheck %s -check-prefix=NEON

; RUN: llc -mtriple=arm-darwin -mcpu=cortex-a8 %s -o - \
; RUN:  | FileCheck %s -check-prefix=NEON

; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a9 %s -o - \
; RUN:  | FileCheck %s -check-prefix=VFP2

define i32 @test1(float %a, float %b) {
; VFP2-LABEL: test1:
; VFP2: vcvt.s32.f32 s{{.}}, s{{.}}
; NEON-LABEL: test1:
; NEON: vadd.f32 [[D0:d[0-9]+]]
; NEON: vcvt.s32.f32 d0, [[D0]]
entry:
        %0 = fadd float %a, %b
        %1 = fptosi float %0 to i32
	ret i32 %1
}

define i32 @test2(float %a, float %b) {
; VFP2-LABEL: test2:
; VFP2: vcvt.u32.f32 s{{.}}, s{{.}}
; NEON-LABEL: test2:
; NEON: vadd.f32 [[D0:d[0-9]+]]
; NEON: vcvt.u32.f32 d0, [[D0]]
entry:
        %0 = fadd float %a, %b
        %1 = fptoui float %0 to i32
	ret i32 %1
}

define float @test3(i32 %a, i32 %b) {
; VFP2-LABEL: test3:
; VFP2: vcvt.f32.u32 s{{.}}, s{{.}}
; NEON-LABEL: test3:
; NEON: vcvt.f32.u32 d
entry:
        %0 = add i32 %a, %b
        %1 = uitofp i32 %0 to float
	ret float %1
}

define float @test4(i32 %a, i32 %b) {
; VFP2-LABEL: test4:
; VFP2: vcvt.f32.s32 s{{.}}, s{{.}}
; NEON-LABEL: test4:
; NEON: vcvt.f32.s32 d
entry:
        %0 = add i32 %a, %b
        %1 = sitofp i32 %0 to float
	ret float %1
}

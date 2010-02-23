; Ensure that floating point operations are lowered to function calls when the
; FPU is not available in the hardware and that function calls are not used
; when the FPU is available in the hardware.
;
; RUN: llc < %s -march=mblaze | FileCheck -check-prefix=FUN %s
; RUN: llc < %s -march=mblaze -mattr=+fpu | FileCheck -check-prefix=FPU %s

define float @test_add(float %a, float %b) {
    ; FUN:        test_add:
    ; FPU:        test_add:

    %tmp.1 = fadd float %a, %b
    ; FUN-NOT:    fadd
    ; FUN:        brlid
    ; FPU-NOT:    brlid
    ; FPU:        fadd

    ret float %tmp.1
    ; FUN:        rtsd
    ; FPU:        rtsd
}

define float @test_sub(float %a, float %b) {
    ; FUN:        test_sub:
    ; FPU:        test_sub:

    %tmp.1 = fsub float %a, %b
    ; FUN-NOT:    frsub
    ; FUN:        brlid
    ; FPU-NOT:    brlid
    ; FPU:        frsub

    ret float %tmp.1
    ; FUN:        rtsd
    ; FPU:        rtsd
}

define float @test_mul(float %a, float %b) {
    ; FUN:        test_mul:
    ; FPU:        test_mul:

    %tmp.1 = fmul float %a, %b
    ; FUN-NOT:    fmul
    ; FUN:        brlid
    ; FPU-NOT:    brlid
    ; FPU:        fmul

    ret float %tmp.1
    ; FUN:        rtsd
    ; FPU:        rtsd
}

define float @test_div(float %a, float %b) {
    ; FUN:        test_div:
    ; FPU:        test_div:

    %tmp.1 = fdiv float %a, %b
    ; FUN-NOT:    fdiv
    ; FUN:        brlid
    ; FPU-NOT:    brlid
    ; FPU:        fdiv

    ret float %tmp.1
    ; FUN:        rtsd
    ; FPU:        rtsd
}

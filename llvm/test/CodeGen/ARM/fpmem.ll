; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s

define float @f1(float %a) {
; CHECK: f1:
; CHECK: mov r0, #0
        ret float 0.000000e+00
}

define float @f2(float* %v, float %u) {
; CHECK: f2:
; CHECK: flds{{.*}}[
        %tmp = load float* %v           ; <float> [#uses=1]
        %tmp1 = fadd float %tmp, %u              ; <float> [#uses=1]
        ret float %tmp1
}

define void @f3(float %a, float %b, float* %v) {
; CHECK: f3:
; CHECK: fsts{{.*}}[
        %tmp = fadd float %a, %b         ; <float> [#uses=1]
        store float %tmp, float* %v
        ret void
}

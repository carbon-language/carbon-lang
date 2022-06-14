; RUN: llc -mtriple=arm -float-abi=soft -mattr=+vfp2 %s -o - | FileCheck %s

define float @f1(float %a) {
; CHECK-LABEL: f1:
; CHECK: mov r0, #0
        ret float 0.000000e+00
}

define float @f2(float* %v, float %u) {
; CHECK-LABEL: f2:
; CHECK: vldr{{.*}}[
        %tmp = load float, float* %v           ; <float> [#uses=1]
        %tmp1 = fadd float %tmp, %u              ; <float> [#uses=1]
        ret float %tmp1
}

define float @f2offset(float* %v, float %u) {
; CHECK-LABEL: f2offset:
; CHECK: vldr{{.*}}, #4]
        %addr = getelementptr float, float* %v, i32 1
        %tmp = load float, float* %addr
        %tmp1 = fadd float %tmp, %u
        ret float %tmp1
}

define float @f2noffset(float* %v, float %u) {
; CHECK-LABEL: f2noffset:
; CHECK: vldr{{.*}}, #-4]
        %addr = getelementptr float, float* %v, i32 -1
        %tmp = load float, float* %addr
        %tmp1 = fadd float %tmp, %u
        ret float %tmp1
}

define void @f3(float %a, float %b, float* %v) {
; CHECK-LABEL: f3:
; CHECK: vstr{{.*}}[
        %tmp = fadd float %a, %b         ; <float> [#uses=1]
        store float %tmp, float* %v
        ret void
}

; RUN: llc < %s -mtriple=arm-linux-gnueabi -mattr=+vfp2 -float-abi=hard | FileCheck %s

define float @f(float %z, double %a, float %b) {
; CHECK: fcpys s0, s1
        %tmp = call float @g(float %b)
        ret float %tmp
}

declare float @g(float)

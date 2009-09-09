; RUN: llc < %s -mtriple=arm-linux-gnueabi -mattr=+vfp2 -float-abi=hard | grep {fcpys s0, s1}

define float @f(float %z, double %a, float %b) {
        %tmp = call float @g(float %b)
        ret float %tmp
}

declare float @g(float)

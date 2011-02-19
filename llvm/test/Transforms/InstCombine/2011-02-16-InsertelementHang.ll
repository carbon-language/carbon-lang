; RUN: opt < %s -instcombine -S | FileCheck %s
; PR9218

%vec2x2 = type { <2 x double>, <2 x double> }

define %vec2x2 @split(double) nounwind alwaysinline {
; CHECK: @split
; CHECK: ret %vec2x2 undef
  %vba = insertelement <2 x double> undef, double %0, i32 2
  ret <2 x double> %vba, <2 x double> %vba
}

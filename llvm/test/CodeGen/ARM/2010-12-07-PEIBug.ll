; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 -mcpu=cortex-a8 | FileCheck %s
; rdar://8728956

define hidden void @foo() nounwind ssp {
entry:
; CHECK: foo:
; CHECK: push {r7, lr}
; CHECK-NEXT: mov r7, sp
; CHECK-NEXT: vpush {d8}
; CHECK-NEXT: vpush {d10, d11}
  %tmp40 = load <4 x i8>* undef
  %tmp41 = extractelement <4 x i8> %tmp40, i32 2
  %conv42 = zext i8 %tmp41 to i32
  %conv43 = sitofp i32 %conv42 to float
  %div44 = fdiv float %conv43, 2.560000e+02
  %vecinit45 = insertelement <4 x float> undef, float %div44, i32 2
  %vecinit46 = insertelement <4 x float> %vecinit45, float 1.000000e+00, i32 3
  store <4 x float> %vecinit46, <4 x float>* undef
  br i1 undef, label %if.then105, label %if.else109

if.then105:                                       ; preds = %entry
  br label %if.end114

if.else109:                                       ; preds = %entry
  br label %if.end114

if.end114:                                        ; preds = %if.else109, %if.then105
  %call185 = call float @bar()
  %vecinit186 = insertelement <4 x float> undef, float %call185, i32 1
  %call189 = call float @bar()
  %vecinit190 = insertelement <4 x float> %vecinit186, float %call189, i32 2
  %vecinit191 = insertelement <4 x float> %vecinit190, float 1.000000e+00, i32 3
  store <4 x float> %vecinit191, <4 x float>* undef
; CHECK: vpop {d10, d11}
; CHECK-NEXT: vpop {d8}
; CHECK-NEXT: pop {r7, pc}
  ret void
}

declare hidden float @bar() nounwind readnone ssp

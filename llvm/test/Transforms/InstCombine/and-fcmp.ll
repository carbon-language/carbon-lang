; RUN: opt < %s -instcombine -S | FileCheck %s

define zeroext i8 @t1(float %x, float %y) nounwind {
       %a = fcmp ueq float %x, %y
       %b = fcmp ord float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
; CHECK: t1
; CHECK: fcmp oeq float %x, %y
; CHECK-NOT: fcmp ueq float %x, %y
; CHECK-NOT: fcmp ord float %x, %y
; CHECK-NOT: and
}

define zeroext i8 @t2(float %x, float %y) nounwind {
       %a = fcmp olt float %x, %y
       %b = fcmp ord float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
; CHECK: t2
; CHECK: fcmp olt float %x, %y
; CHECK-NOT: fcmp ord float %x, %y
; CHECK-NOT: and
}

define zeroext i8 @t3(float %x, float %y) nounwind {
       %a = fcmp oge float %x, %y
       %b = fcmp uno float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
; CHECK: t3
; CHECK: ret i8 0
}

define zeroext i8 @t4(float %x, float %y) nounwind {
       %a = fcmp one float %y, %x
       %b = fcmp ord float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
; CHECK: t4
; CHECK: fcmp one float %y, %x
; CHECK-NOT: fcmp ord float %x, %y
; CHECK-NOT: and
}

define zeroext i8 @t5(float %x, float %y) nounwind {
       %a = fcmp ord float %x, %y
       %b = fcmp uno float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
; CHECK: t5
; CHECK: ret i8 0
}

define zeroext i8 @t6(float %x, float %y) nounwind {
       %a = fcmp uno float %x, %y
       %b = fcmp ord float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
; CHECK: t6
; CHECK: ret i8 0
}

define zeroext i8 @t7(float %x, float %y) nounwind {
       %a = fcmp uno float %x, %y
       %b = fcmp ult float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
; CHECK: t7
; CHECK: fcmp uno
; CHECK-NOT: fcmp ult
}

; PR15737
define i1 @t8(float %a, double %b) {
  %cmp = fcmp ord float %a, 0.000000e+00
  %cmp1 = fcmp ord double %b, 0.000000e+00
  %and = and i1 %cmp, %cmp1
  ret i1 %and
; CHECK: t8
; CHECK: fcmp ord
; CHECK: fcmp ord
}

define <2 x i1> @t9(<2 x float> %a, <2 x double> %b) {
  %cmp = fcmp ord <2 x float> %a, zeroinitializer
  %cmp1 = fcmp ord <2 x double> %b, zeroinitializer
  %and = and <2 x i1> %cmp, %cmp1
  ret <2 x i1> %and
; CHECK: t9
; CHECK: fcmp ord
; CHECK: fcmp ord
}

; RUN: opt < %s -instcombine -S | FileCheck %s

; PR1738
define i1 @test1(double %X, double %Y) {
        %tmp9 = fcmp ord double %X, 0.000000e+00
        %tmp13 = fcmp ord double %Y, 0.000000e+00
        %bothcond = and i1 %tmp13, %tmp9
        ret i1 %bothcond
; CHECK:  fcmp ord double %Y, %X
}

define i1 @test2(i1 %X, i1 %Y) {
  %a = and i1 %X, %Y
  %b = and i1 %a, %X
  ret i1 %b
; CHECK: @test2
; CHECK-NEXT: and i1 %X, %Y
; CHECK-NEXT: ret
}

define i32 @test3(i32 %X, i32 %Y) {
  %a = and i32 %X, %Y
  %b = and i32 %Y, %a
  ret i32 %b
; CHECK: @test3
; CHECK-NEXT: and i32 %X, %Y
; CHECK-NEXT: ret
}

define i1 @test4(i32 %X) {
  %a = icmp ult i32 %X, 31
  %b = icmp slt i32 %X, 0
  %c = and i1 %a, %b
  ret i1 %c
; CHECK: @test4
; CHECK-NEXT: ret i1 false
}

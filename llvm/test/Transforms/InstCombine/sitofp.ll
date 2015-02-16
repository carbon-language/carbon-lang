; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK-LABEL: test1
; CHECK: ret i1 true
define i1 @test1(i8 %A) {
  %B = sitofp i8 %A to double
  %C = fcmp ult double %B, 128.0
  ret i1 %C
}

; CHECK-LABEL: test2
; CHECK: ret i1 true
define i1 @test2(i8 %A) {
  %B = sitofp i8 %A to double
  %C = fcmp ugt double %B, -128.1
  ret i1 %C
}

; CHECK-LABEL: test3
; CHECK: ret i1 true
define i1 @test3(i8 %A) {
  %B = sitofp i8 %A to double
  %C = fcmp ule double %B, 127.0
  ret i1 %C
}

; CHECK-LABEL: test4
; CHECK: icmp ne i8 %A, 127
; CHECK-NEXT: ret i1
define i1 @test4(i8 %A) {
  %B = sitofp i8 %A to double
  %C = fcmp ult double %B, 127.0
  ret i1 %C
}

; CHECK-LABEL: test5
; CHECK: ret i32
define i32 @test5(i32 %A) {
  %B = sitofp i32 %A to double
  %C = fptosi double %B to i32
  %D = uitofp i32 %C to double
  %E = fptoui double %D to i32
  ret i32 %E
}

; CHECK-LABEL: test6
; CHECK: and i32 %A, 39
; CHECK-NEXT: ret i32
define i32 @test6(i32 %A) {
  %B = and i32 %A, 7
  %C = and i32 %A, 32
  %D = sitofp i32 %B to double
  %E = sitofp i32 %C to double
  %F = fadd double %D, %E
  %G = fptosi double %F to i32
  ret i32 %G
}

; CHECK-LABEL: test7
; CHECK: ret i32
define i32 @test7(i32 %A) nounwind {
  %B = sitofp i32 %A to double
  %C = fptoui double %B to i32
  ret i32 %C
}

; CHECK-LABEL: test8
; CHECK: ret i32
define i32 @test8(i32 %A) nounwind {
  %B = uitofp i32 %A to double
  %C = fptosi double %B to i32
  ret i32 %C
}


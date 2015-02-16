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

; CHECK-LABEL: test9
; CHECK: zext i8
; CHECK-NEXT: ret i32
define i32 @test9(i8 %A) nounwind {
  %B = sitofp i8 %A to float
  %C = fptoui float %B to i32
  ret i32 %C
}

; CHECK-LABEL: test10
; CHECK: sext i8
; CHECK-NEXT: ret i32
define i32 @test10(i8 %A) nounwind {
  %B = sitofp i8 %A to float
  %C = fptosi float %B to i32
  ret i32 %C
}

; If the input value is outside of the range of the output cast, it's
; undefined behavior, so we can assume it fits.
; CHECK-LABEL: test11
; CHECK: trunc
; CHECK-NEXT: ret i8
define i8 @test11(i32 %A) nounwind {
  %B = sitofp i32 %A to float
  %C = fptosi float %B to i8
  ret i8 %C
}

; If the input value is negative, it'll be outside the range of the
; output cast, and thus undefined behavior.
; CHECK-LABEL: test12
; CHECK: zext i8
; CHECK-NEXT: ret i32
define i32 @test12(i8 %A) nounwind {
  %B = sitofp i8 %A to float
  %C = fptoui float %B to i32
  ret i32 %C
}

; This can't fold because the 25-bit input doesn't fit in the mantissa.
; CHECK-LABEL: test13
; CHECK: uitofp
; CHECK-NEXT: fptoui
define i32 @test13(i25 %A) nounwind {
  %B = uitofp i25 %A to float
  %C = fptoui float %B to i32
  ret i32 %C
}

; But this one can.
; CHECK-LABEL: test14
; CHECK: zext i24
; CHECK-NEXT: ret i32
define i32 @test14(i24 %A) nounwind {
  %B = uitofp i24 %A to float
  %C = fptoui float %B to i32
  ret i32 %C
}

; And this one can too.
; CHECK-LABEL: test15
; CHECK: trunc i32
; CHECK-NEXT: ret i24
define i24 @test15(i32 %A) nounwind {
  %B = uitofp i32 %A to float
  %C = fptoui float %B to i24
  ret i24 %C
}

; This can fold because the 25-bit input is signed and we disard the sign bit.
; CHECK-LABEL: test16
; CHECK: zext
define i32 @test16(i25 %A) nounwind {
 %B = sitofp i25 %A to float
 %C = fptoui float %B to i32
 ret i32 %C
}

; This can't fold because the 26-bit input won't fit the mantissa
; even after disarding the signed bit.
; CHECK-LABEL: test17
; CHECK: sitofp
; CHECK-NEXT: fptoui
define i32 @test17(i26 %A) nounwind {
 %B = sitofp i26 %A to float
 %C = fptoui float %B to i32
 ret i32 %C
}

; This can fold because the 54-bit output is signed and we disard the sign bit.
; CHECK-LABEL: test18
; CHECK: trunc
define i54 @test18(i64 %A) nounwind {
 %B = sitofp i64 %A to double
 %C = fptosi double %B to i54
 ret i54 %C
}

; This can't fold because the 55-bit output won't fit the mantissa
; even after disarding the sign bit.
; CHECK-LABEL: test19
; CHECK: sitofp
; CHECK-NEXT: fptosi
define i55 @test19(i64 %A) nounwind {
 %B = sitofp i64 %A to double
 %C = fptosi double %B to i55
 ret i55 %C
}


; RUN: opt < %s -instcombine -S | FileCheck %s

@G16 = internal constant [10 x i16] [i16 35, i16 82, i16 69, i16 81, i16 85, 
                                     i16 73, i16 82, i16 69, i16 68, i16 0]
@GD = internal constant [3 x double] [double 1.0, double 4.0, double -20.0]

define i1 @test1(i32 %X) {
  %P = getelementptr [10 x i16]* @G16, i32 0, i32 %X
  %Q = load i16* %P
  %R = icmp eq i16 %Q, 0
  ret i1 %R
; CHECK: @test1
; CHECK-NEXT: %R = icmp eq i32 %X, 9
; CHECK-NEXT: ret i1 %R
}

define i1 @test2(i32 %X) {
  %P = getelementptr [10 x i16]* @G16, i32 0, i32 %X
  %Q = load i16* %P
  %R = icmp slt i16 %Q, 85
  ret i1 %R
; CHECK: @test2
; CHECK-NEXT: %R = icmp ne i32 %X, 4
; CHECK-NEXT: ret i1 %R
}

define i1 @test3(i32 %X) {
  %P = getelementptr [3 x double]* @GD, i32 0, i32 %X
  %Q = load double* %P
  %R = fcmp oeq double %Q, 1.0
  ret i1 %R
; CHECK: @test3
; CHECK-NEXT: %R = icmp eq i32 %X, 0
; CHECK-NEXT: ret i1 %R
}


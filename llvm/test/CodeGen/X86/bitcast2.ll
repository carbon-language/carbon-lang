; RUN: llc < %s -march=x86-64 | FileCheck %s
; RUN: llc < %s -march=x86-64 | not grep rsp

define i64 @test1(double %A) {
; CHECK: test1
; CHECK: movq
   %B = bitcast double %A to i64
   ret i64 %B
}

define double @test2(i64 %A) {
; CHECK: test2
; CHECK: movq
   %B = bitcast i64 %A to double
   ret double %B
}

define i32 @test3(float %A) {
; CHECK: test3
; CHECK: movd
   %B = bitcast float %A to i32
   ret i32 %B
}

define float @test4(i32 %A) {
; CHECK: test4
; CHECK: movd
   %B = bitcast i32 %A to float
   ret float %B
}


; RUN: llc < %s -march=x86-64 -mcpu=core2 -mattr=+sse2 | FileCheck %s


define double @test1(double %A) {
  %1 = bitcast double %A to <2 x i32>
  %add = add <2 x i32> %1, <i32 3, i32 5>
  %2 = bitcast <2 x i32> %add to double
  ret double %2
}
; FIXME: Ideally we should be able to fold the entire body of @test1 into a
; single paddd instruction. At the moment we produce the sequence 
; pshufd+paddq+pshufd.

; CHECK-LABEL: test1
; CHECK-NOT: movsd
; CHECK: pshufd
; CHECK-NEXT: paddq
; CHECK-NEXT: pshufd
; CHECK-NEXT: ret


define double @test2(double %A, double %B) {
  %1 = bitcast double %A to <2 x i32>
  %2 = bitcast double %B to <2 x i32>
  %add = add <2 x i32> %1, %2
  %3 = bitcast <2 x i32> %add to double
  ret double %3
}
; FIXME: Ideally we should be able to fold the entire body of @test2 into a
; single 'paddd %xmm1, %xmm0' instruction. At the moment we produce the
; sequence pshufd+pshufd+paddq+pshufd.

; CHECK-LABEL: test2
; CHECK-NOT: movsd
; CHECK: pshufd
; CHECK-NEXT: pshufd
; CHECK-NEXT: paddq
; CHECK-NEXT: pshufd
; CHECK-NEXT: ret


define i64 @test3(i64 %A) {
  %1 = bitcast i64 %A to <2 x float>
  %add = fadd <2 x float> %1, <float 3.0, float 5.0>
  %2 = bitcast <2 x float> %add to i64
  ret i64 %2
}
; CHECK-LABEL: test3
; CHECK-NOT: pshufd
; CHECK: addps
; CHECK-NOT: pshufd
; CHECK: ret


define i64 @test4(i64 %A) {
  %1 = bitcast i64 %A to <2 x i32>
  %add = add <2 x i32> %1, <i32 3, i32 5>
  %2 = bitcast <2 x i32> %add to i64
  ret i64 %2
}
; FIXME: At the moment we still produce the sequence pshufd+paddq+pshufd.
; Ideally, we should fold that sequence into a single paddd.

; CHECK-LABEL: test4
; CHECK: pshufd
; CHECK-NEXT: paddq
; CHECK-NEXT: pshufd
; CHECK: ret


define double @test5(double %A) {
  %1 = bitcast double %A to <2 x float>
  %add = fadd <2 x float> %1, <float 3.0, float 5.0>
  %2 = bitcast <2 x float> %add to double
  ret double %2
}
; CHECK-LABEL: test5
; CHECK: addps
; CHECK-NEXT: ret


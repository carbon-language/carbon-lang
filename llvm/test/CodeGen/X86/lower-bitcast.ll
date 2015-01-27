; RUN: llc < %s -march=x86-64 -mcpu=core2 -mattr=+sse2 | FileCheck %s
; RUN: llc < %s -march=x86-64 -mcpu=core2 -mattr=+sse2 -x86-experimental-vector-widening-legalization | FileCheck %s --check-prefix=CHECK-WIDE


define double @test1(double %A) {
  %1 = bitcast double %A to <2 x i32>
  %add = add <2 x i32> %1, <i32 3, i32 5>
  %2 = bitcast <2 x i32> %add to double
  ret double %2
}
; FIXME: Ideally we should be able to fold the entire body of @test1 into a
; single paddd instruction. At the moment we produce the sequence 
; pshufd+paddq+pshufd. This is fixed with the widening legalization.
;
; CHECK-LABEL: test1
; CHECK-NOT: movsd
; CHECK: pshufd
; CHECK-NEXT: paddd
; CHECK-NEXT: pshufd
; CHECK-NEXT: ret
;
; CHECK-WIDE-LABEL: test1
; CHECK-WIDE-NOT: movsd
; CHECK-WIDE: paddd
; CHECK-WIDE-NEXT: ret


define double @test2(double %A, double %B) {
  %1 = bitcast double %A to <2 x i32>
  %2 = bitcast double %B to <2 x i32>
  %add = add <2 x i32> %1, %2
  %3 = bitcast <2 x i32> %add to double
  ret double %3
}
; CHECK-LABEL: test2
; CHECK-NOT: movsd
; CHECK: paddd
; CHECK-NEXT: ret
;
; CHECK-WIDE-LABEL: test2
; CHECK-WIDE-NOT: movsd
; CHECK-WIDE: paddd
; CHECK-WIDE-NEXT: ret


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
;
; CHECK-WIDE-LABEL: test3
; CHECK-WIDE-NOT: pshufd
; CHECK-WIDE: addps
; CHECK-WIDE-NOT: pshufd
; CHECK-WIDE: ret


define i64 @test4(i64 %A) {
  %1 = bitcast i64 %A to <2 x i32>
  %add = add <2 x i32> %1, <i32 3, i32 5>
  %2 = bitcast <2 x i32> %add to i64
  ret i64 %2
}
; FIXME: At the moment we still produce the sequence pshufd+paddd+pshufd.
; Ideally, we should fold that sequence into a single paddd. This is fixed with
; the widening legalization.
;
; CHECK-LABEL: test4
; CHECK: pshufd
; CHECK-NEXT: paddd
; CHECK-NEXT: pshufd
; CHECK: ret
;
; CHECK-WIDE-LABEL: test4
; CHECK-WIDE: movd %{{rdi|rcx}},
; CHECK-WIDE-NEXT: paddd
; CHECK-WIDE-NEXT: movd {{.*}}, %rax
; CHECK-WIDE: ret


define double @test5(double %A) {
  %1 = bitcast double %A to <2 x float>
  %add = fadd <2 x float> %1, <float 3.0, float 5.0>
  %2 = bitcast <2 x float> %add to double
  ret double %2
}
; CHECK-LABEL: test5
; CHECK: addps
; CHECK-NEXT: ret
;
; CHECK-WIDE-LABEL: test5
; CHECK-WIDE: addps
; CHECK-WIDE-NEXT: ret


define double @test6(double %A) {
  %1 = bitcast double %A to <4 x i16>
  %add = add <4 x i16> %1, <i16 3, i16 4, i16 5, i16 6>
  %2 = bitcast <4 x i16> %add to double
  ret double %2
}
; FIXME: Ideally we should be able to fold the entire body of @test6 into a
; single paddw instruction. This is fixed with the widening legalization.
;
; CHECK-LABEL: test6
; CHECK-NOT: movsd
; CHECK: punpcklwd
; CHECK-NEXT: paddw
; CHECK-NEXT: pshufb
; CHECK-NEXT: ret
;
; CHECK-WIDE-LABEL: test6
; CHECK-WIDE-NOT: mov
; CHECK-WIDE-NOT: punpcklwd
; CHECK-WIDE: paddw
; CHECK-WIDE-NEXT: ret


define double @test7(double %A, double %B) {
  %1 = bitcast double %A to <4 x i16>
  %2 = bitcast double %B to <4 x i16>
  %add = add <4 x i16> %1, %2
  %3 = bitcast <4 x i16> %add to double
  ret double %3
}
; CHECK-LABEL: test7
; CHECK-NOT: movsd
; CHECK-NOT: punpcklwd
; CHECK: paddw
; CHECK-NEXT: ret
;
; CHECK-WIDE-LABEL: test7
; CHECK-WIDE-NOT: movsd
; CHECK-WIDE-NOT: punpcklwd
; CHECK-WIDE: paddw
; CHECK-WIDE-NEXT: ret


define double @test8(double %A) {
  %1 = bitcast double %A to <8 x i8>
  %add = add <8 x i8> %1, <i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10>
  %2 = bitcast <8 x i8> %add to double
  ret double %2
}
; FIXME: Ideally we should be able to fold the entire body of @test8 into a
; single paddb instruction. At the moment we produce the sequence 
; pshufd+paddw+pshufd. This is fixed with the widening legalization.
;
; CHECK-LABEL: test8
; CHECK-NOT: movsd
; CHECK: punpcklbw
; CHECK-NEXT: paddb
; CHECK-NEXT: pshufb
; CHECK-NEXT: ret
;
; CHECK-WIDE-LABEL: test8
; CHECK-WIDE-NOT: movsd
; CHECK-WIDE-NOT: punpcklbw
; CHECK-WIDE: paddb
; CHECK-WIDE-NEXT: ret


define double @test9(double %A, double %B) {
  %1 = bitcast double %A to <8 x i8>
  %2 = bitcast double %B to <8 x i8>
  %add = add <8 x i8> %1, %2
  %3 = bitcast <8 x i8> %add to double
  ret double %3
}
; CHECK-LABEL: test9
; CHECK-NOT: movsd
; CHECK-NOT: punpcklbw
; CHECK: paddb
; CHECK-NEXT: ret
;
; CHECK-WIDE-LABEL: test9
; CHECK-WIDE-NOT: movsd
; CHECK-WIDE-NOT: punpcklbw
; CHECK-WIDE: paddb
; CHECK-WIDE-NEXT: ret


; RUN: opt %s -instcombine -S | FileCheck %s

%overflow.result = type {i8, i1}

declare %overflow.result @llvm.uadd.with.overflow.i8(i8, i8)
declare %overflow.result @llvm.umul.with.overflow.i8(i8, i8)
declare double @llvm.powi.f64(double, i32) nounwind readonly
declare i32 @llvm.cttz.i32(i32) nounwind readnone
declare i32 @llvm.ctlz.i32(i32) nounwind readnone
declare i32 @llvm.ctpop.i32(i32) nounwind readnone
declare i8 @llvm.ctlz.i8(i8) nounwind readnone

define i8 @test1(i8 %A, i8 %B) {
  %x = call %overflow.result @llvm.uadd.with.overflow.i8(i8 %A, i8 %B)
  %y = extractvalue %overflow.result %x, 0
  ret i8 %y
; CHECK: @test1
; CHECK-NEXT: %y = add i8 %A, %B
; CHECK-NEXT: ret i8 %y
}

define i8 @test2(i8 %A, i8 %B, i1* %overflowPtr) {
  %and.A = and i8 %A, 127
  %and.B = and i8 %B, 127
  %x = call %overflow.result @llvm.uadd.with.overflow.i8(i8 %and.A, i8 %and.B)
  %y = extractvalue %overflow.result %x, 0
  %z = extractvalue %overflow.result %x, 1
  store i1 %z, i1* %overflowPtr
  ret i8 %y
; CHECK: @test2
; CHECK-NEXT: %and.A = and i8 %A, 127
; CHECK-NEXT: %and.B = and i8 %B, 127
; CHECK-NEXT: %1 = add nuw i8 %and.A, %and.B
; CHECK-NEXT: store i1 false, i1* %overflowPtr
; CHECK-NEXT: ret i8 %1
}

define i8 @test3(i8 %A, i8 %B, i1* %overflowPtr) {
  %or.A = or i8 %A, -128
  %or.B = or i8 %B, -128
  %x = call %overflow.result @llvm.uadd.with.overflow.i8(i8 %or.A, i8 %or.B)
  %y = extractvalue %overflow.result %x, 0
  %z = extractvalue %overflow.result %x, 1
  store i1 %z, i1* %overflowPtr
  ret i8 %y
; CHECK: @test3
; CHECK-NEXT: %or.A = or i8 %A, -128
; CHECK-NEXT: %or.B = or i8 %B, -128
; CHECK-NEXT: %1 = add i8 %or.A, %or.B
; CHECK-NEXT: store i1 true, i1* %overflowPtr
; CHECK-NEXT: ret i8 %1
}

define i8 @test4(i8 %A, i1* %overflowPtr) {
  %x = call %overflow.result @llvm.uadd.with.overflow.i8(i8 undef, i8 %A)
  %y = extractvalue %overflow.result %x, 0
  %z = extractvalue %overflow.result %x, 1
  store i1 %z, i1* %overflowPtr
  ret i8 %y
; CHECK: @test4
; CHECK-NEXT: ret i8 undef
}

define i8 @test5(i8 %A, i1* %overflowPtr) {
  %x = call %overflow.result @llvm.umul.with.overflow.i8(i8 0, i8 %A)
  %y = extractvalue %overflow.result %x, 0
  %z = extractvalue %overflow.result %x, 1
  store i1 %z, i1* %overflowPtr
  ret i8 %y
; CHECK: @test5
; CHECK-NEXT: store i1 false, i1* %overflowPtr
; CHECK-NEXT: ret i8 0
}

define i8 @test6(i8 %A, i1* %overflowPtr) {
  %x = call %overflow.result @llvm.umul.with.overflow.i8(i8 1, i8 %A)
  %y = extractvalue %overflow.result %x, 0
  %z = extractvalue %overflow.result %x, 1
  store i1 %z, i1* %overflowPtr
  ret i8 %y
; CHECK: @test6
; CHECK-NEXT: store i1 false, i1* %overflowPtr
; CHECK-NEXT: ret i8 %A
}

define void @powi(double %V, double *%P) {
entry:
  %A = tail call double @llvm.powi.f64(double %V, i32 -1) nounwind
  volatile store double %A, double* %P

  %B = tail call double @llvm.powi.f64(double %V, i32 0) nounwind
  volatile store double %B, double* %P

  %C = tail call double @llvm.powi.f64(double %V, i32 1) nounwind
  volatile store double %C, double* %P
  ret void
; CHECK: @powi
; CHECK: %A = fdiv double 1.0{{.*}}, %V
; CHECK: volatile store double %A, 
; CHECK: volatile store double 1.0 
; CHECK: volatile store double %V
}

define i32 @cttz(i32 %a) {
entry:
  %or = or i32 %a, 8
  %and = and i32 %or, -8
  %count = tail call i32 @llvm.cttz.i32(i32 %and) nounwind readnone
  ret i32 %count
; CHECK: @cttz
; CHECK-NEXT: entry:
; CHECK-NEXT: ret i32 3
}

define i8 @ctlz(i8 %a) {
entry:
  %or = or i8 %a, 32
  %and = and i8 %or, 63
  %count = tail call i8 @llvm.ctlz.i8(i8 %and) nounwind readnone
  ret i8 %count
; CHECK: @ctlz
; CHECK-NEXT: entry:
; CHECK-NEXT: ret i8 2
}

define void @cmp.simplify(i32 %a, i32 %b, i1* %c) {
entry:
  %lz = tail call i32 @llvm.ctlz.i32(i32 %a) nounwind readnone
  %lz.cmp = icmp eq i32 %lz, 32
  volatile store i1 %lz.cmp, i1* %c
  %tz = tail call i32 @llvm.cttz.i32(i32 %a) nounwind readnone
  %tz.cmp = icmp ne i32 %tz, 32
  volatile store i1 %tz.cmp, i1* %c
  %pop = tail call i32 @llvm.ctpop.i32(i32 %b) nounwind readnone
  %pop.cmp = icmp eq i32 %pop, 0
  volatile store i1 %pop.cmp, i1* %c
  ret void
; CHECK: @cmp.simplify
; CHECK-NEXT: entry:
; CHECK-NEXT: %lz.cmp = icmp eq i32 %a, 0
; CHECK-NEXT: volatile store i1 %lz.cmp, i1* %c
; CHECK-NEXT: %tz.cmp = icmp ne i32 %a, 0
; CHECK-NEXT: volatile store i1 %tz.cmp, i1* %c
; CHECK-NEXT: %pop.cmp = icmp eq i32 %b, 0
; CHECK-NEXT: volatile store i1 %pop.cmp, i1* %c
}


define i32 @cttz_simplify1(i32 %x) nounwind readnone ssp {
  %tmp1 = tail call i32 @llvm.ctlz.i32(i32 %x)    ; <i32> [#uses=1]
  %shr3 = lshr i32 %tmp1, 5                       ; <i32> [#uses=1]
  ret i32 %shr3
  
; CHECK: @cttz_simplify1
; CHECK: icmp eq i32 %x, 0
; CHECK-NEXT: zext i1 
; CHECK-NEXT: ret i32
}

declare i32 @llvm.ctlz.i32(i32) nounwind readnone


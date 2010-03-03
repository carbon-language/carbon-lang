; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @test1(i32 %X) {
entry:
        icmp slt i32 %X, 0              ; <i1>:0 [#uses=1]
        zext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
; CHECK: @test1
; CHECK: lshr i32 %X, 31
; CHECK-NEXT: ret i32
}

define i32 @test2(i32 %X) {
entry:
        icmp ult i32 %X, -2147483648            ; <i1>:0 [#uses=1]
        zext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
; CHECK: @test2
; CHECK: lshr i32 %X, 31
; CHECK-NEXT: xor i32
; CHECK-NEXT: ret i32
}

define i32 @test3(i32 %X) {
entry:
        icmp slt i32 %X, 0              ; <i1>:0 [#uses=1]
        sext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
; CHECK: @test3
; CHECK: ashr i32 %X, 31
; CHECK-NEXT: ret i32
}

define i32 @test4(i32 %X) {
entry:
        icmp ult i32 %X, -2147483648            ; <i1>:0 [#uses=1]
        sext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
; CHECK: @test4
; CHECK: ashr i32 %X, 31
; CHECK-NEXT: xor i32
; CHECK-NEXT: ret i32
}

; PR4837
define <2 x i1> @test5(<2 x i64> %x) {
entry:
  %V = icmp eq <2 x i64> %x, undef
  ret <2 x i1> %V
; CHECK: @test5
; CHECK: ret <2 x i1> <i1 true, i1 true>
}

define i32 @test6(i32 %a, i32 %b) {
        %c = icmp sle i32 %a, -1
        %d = zext i1 %c to i32
        %e = sub i32 0, %d
        %f = and i32 %e, %b
        ret i32 %f
; CHECK: @test6
; CHECK-NEXT: ashr i32 %a, 31
; CHECK-NEXT: %f = and i32 %e, %b
; CHECK-NEXT: ret i32 %f
}


define i1 @test7(i32 %x) {
entry:
  %a = add i32 %x, -1
  %b = icmp ult i32 %a, %x
  ret i1 %b
; CHECK: @test7
; CHECK: %b = icmp ne i32 %x, 0
; CHECK: ret i1 %b
}

define i1 @test8(i32 %x){
entry:
  %a = add i32 %x, -1 
  %b = icmp eq i32 %a, %x
  ret i1 %b
; CHECK: @test8
; CHECK: ret i1 false
}

define i1 @test9(i32 %x)  {
entry:
  %a = add i32 %x, -2
  %b = icmp ugt i32 %x, %a 
  ret i1 %b
; CHECK: @test9
; CHECK: icmp ugt i32 %x, 1
; CHECK: ret i1 %b
}

define i1 @test10(i32 %x){
entry:
  %a = add i32 %x, -1      
  %b = icmp slt i32 %a, %x 
  ret i1 %b
  
; CHECK: @test10
; CHECK: %b = icmp ne i32 %x, -2147483648
; CHECK: ret i1 %b
}

define i1 @test11(i32 %x) {
  %a = add nsw i32 %x, 8
  %b = icmp slt i32 %x, %a
  ret i1 %b
; CHECK: @test11  
; CHECK: ret i1 true
}

; PR6195
define i1 @test12(i1 %A) {
  %S = select i1 %A, i64 -4294967295, i64 8589934591
  %B = icmp ne i64 bitcast (<2 x i32> <i32 1, i32 -1> to i64), %S
  ret i1 %B
; CHECK: @test12
; CHECK-NEXT: %B = select i1
; CHECK-NEXT: ret i1 %B
}

; PR6481
define i1 @test13(i8 %X) nounwind readnone {
entry:
        %cmp = icmp slt i8 undef, %X
        ret i1 %cmp
; CHECK: @test13
; CHECK: ret i1 false
}


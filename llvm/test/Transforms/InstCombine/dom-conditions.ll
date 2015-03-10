; RUN: opt -instcombine -value-tracking-dom-conditions=1 -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-p1:16:16:16-p2:32:32:32-p3:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define i1 @test_cmp_ult(i64 %A) {
; CHECK-LABEL: @test_cmp_ult
entry:
  %cmp = icmp ult i64 %A, 64
  br i1 %cmp, label %taken, label %untaken

taken:
; CHECK-LABEL: taken:
; CHECK-NEXT: ret i1 false
  %cmp2 = icmp ugt i64 %A, 64
  ret i1 %cmp2
untaken:
  ret i1 true
}

define i1 @test_cmp_ule(i64 %A) {
; CHECK-LABEL: @test_cmp_ule
entry:
  %cmp = icmp ule i64 %A, 64
  br i1 %cmp, label %taken, label %untaken

taken:
; CHECK-LABEL: taken:
; CHECK-NEXT: ret i1 false
  %cmp2 = icmp ugt i64 %A, 128
  ret i1 %cmp2
untaken:
  ret i1 true
}

define i1 @test_cmp_sgt(i32 %A) {
; CHECK-LABEL: @test_cmp_sgt
entry:
  %cmp = icmp sgt i32 %A, 10
  br i1 %cmp, label %taken, label %untaken

taken:
; CHECK-LABEL: taken:
; CHECK-NEXT: ret i1 true
  %cmp2 = icmp sgt i32 %A, -1
  ret i1 %cmp2
untaken:
  ret i1 true
}

define i64 @test_add_zero_bits(i64 %A) {
; CHECK-LABEL: @test_add_zero_bits
entry:
  %cmp = icmp eq i64 %A, 2
  br i1 %cmp, label %taken, label %untaken

taken:
; CHECK-LABEL: taken:
; CHECK-NEXT: ret i64 3
  %add = add i64 %A, 1
  ret i64 %add
untaken:
  ret i64 %A
}

define i64 @test_add_nsw(i64 %A) {
; CHECK-LABEL: @test_add_nsw
entry:
  %cmp = icmp ult i64 %A, 20
  br i1 %cmp, label %taken, label %untaken

taken:
; CHECK-LABEL: taken:
; CHECK-NEXT: %add = add nuw nsw i64 %A, 1
; CHECK-NEXT: ret i64 %add
  %add = add i64 %A, 1
  ret i64 %add
untaken:
  ret i64 %A
}

; After sinking the instructions into the if block, check that we 
; can simplify some of them using dominating conditions.
define i32 @test_add_zero_bits_sink(i32 %x) nounwind ssp {
; CHECK-LABEL: @test_add_zero_bits_sink(
; CHECK-NOT: sdiv i32
entry:
  %a = add nsw i32 %x, 16
  %b = sdiv i32 %a, %x
  %cmp = icmp ult i32 %x, 7
  br i1 %cmp, label %bb1, label %bb2

bb1:
; CHECK-LABEL: bb1:
; CHECK-NEXT: or i32 %x, 16
; CHECK-NEXT: udiv i32
  ret i32 %b

bb2:
  ret i32 %x
}

; A condition in the same block gives no information
define i32 @test_neg1(i32 %x) nounwind ssp {
; CHECK-LABEL: @test_neg1
; CHECK: add
; CHECK: sdiv
; CHECK: icmp
; CHECK: select
entry:
  %a = add nsw i32 %x, 16
  %b = sdiv i32 %a, %x
  %cmp = icmp ult i32 %x, 7
  %ret = select i1 %cmp, i32 %a, i32 %b
  ret i32 %ret
}

; A non-dominating edge gives no information 
define i32 @test_neg2(i32 %x) {
; CHECK-LABEL: @test_neg2
entry:
  %cmp = icmp ult i32 %x, 7
  br i1 %cmp, label %bb1, label %merge

bb1:
  br label %merge

merge:
; CHECK-LABEL: merge:
; CHECK: icmp
; CHECK: select
  %cmp2 = icmp ult i32 %x, 7
  %ret = select i1 %cmp2, i32 %x, i32 0
  ret i32 %ret
}

; A unconditional branch expressed as a condition one gives no
; information (and shouldn't trip any asserts.)
define i32 @test_neg3(i32 %x) {
; CHECK-LABEL: @test_neg3
entry:
  %cmp = icmp ult i32 %x, 7
  br i1 %cmp, label %merge, label %merge
merge:
; CHECK-LABEL: merge:
; CHECK: icmp
; CHECK: select
  %cmp2 = icmp ult i32 %x, 7
  %ret = select i1 %cmp2, i32 %x, i32 0
  ret i32 %ret
}

declare i32 @bar()

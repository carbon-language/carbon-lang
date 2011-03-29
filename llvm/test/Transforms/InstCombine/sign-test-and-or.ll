; RUN: opt -S -instcombine < %s | FileCheck %s

declare void @foo()

define void @test1(i32 %a, i32 %b) nounwind {
  %1 = icmp slt i32 %a, 0
  %2 = icmp slt i32 %b, 0
  %or.cond = or i1 %1, %2
  br i1 %or.cond, label %if.then, label %if.end

; CHECK: @test1
; CHECK-NEXT: %1 = or i32 %a, %b
; CHECK-NEXT: %2 = icmp slt i32 %1, 0
; CHECK-NEXT: br

if.then:
  tail call void @foo() nounwind
  ret void

if.end:
  ret void
}

define void @test2(i32 %a, i32 %b) nounwind {
  %1 = icmp sgt i32 %a, -1
  %2 = icmp sgt i32 %b, -1
  %or.cond = or i1 %1, %2
  br i1 %or.cond, label %if.then, label %if.end

; CHECK: @test2
; CHECK-NEXT: %1 = and i32 %a, %b
; CHECK-NEXT: %2 = icmp sgt i32 %1, -1
; CHECK-NEXT: br

if.then:
  tail call void @foo() nounwind
  ret void

if.end:
  ret void
}

define void @test3(i32 %a, i32 %b) nounwind {
  %1 = icmp slt i32 %a, 0
  %2 = icmp slt i32 %b, 0
  %or.cond = and i1 %1, %2
  br i1 %or.cond, label %if.then, label %if.end

; CHECK: @test3
; CHECK-NEXT: %1 = and i32 %a, %b
; CHECK-NEXT: %2 = icmp slt i32 %1, 0
; CHECK-NEXT: br

if.then:
  tail call void @foo() nounwind
  ret void

if.end:
  ret void
}

define void @test4(i32 %a, i32 %b) nounwind {
  %1 = icmp sgt i32 %a, -1
  %2 = icmp sgt i32 %b, -1
  %or.cond = and i1 %1, %2
  br i1 %or.cond, label %if.then, label %if.end

; CHECK: @test4
; CHECK-NEXT: %1 = or i32 %a, %b
; CHECK-NEXT: %2 = icmp sgt i32 %1, -1
; CHECK-NEXT: br

if.then:
  tail call void @foo() nounwind
  ret void

if.end:
  ret void
}

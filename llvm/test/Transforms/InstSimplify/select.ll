; RUN: opt < %s -instsimplify -S | FileCheck %s

define i32 @test1(i32 %x) {
  %and = and i32 %x, 1
  %cmp = icmp eq i32 %and, 0
  %and1 = and i32 %x, -2
  %and1.x = select i1 %cmp, i32 %and1, i32 %x
  ret i32 %and1.x
; CHECK-LABEL: @test1(
; CHECK: ret i32 %x
}

define i32 @test2(i32 %x) {
  %and = and i32 %x, 1
  %cmp = icmp ne i32 %and, 0
  %and1 = and i32 %x, -2
  %and1.x = select i1 %cmp, i32 %x, i32 %and1
  ret i32 %and1.x
; CHECK-LABEL: @test2(
; CHECK: ret i32 %x
}

define i32 @test3(i32 %x) {
  %and = and i32 %x, 1
  %cmp = icmp ne i32 %and, 0
  %and1 = and i32 %x, -2
  %and1.x = select i1 %cmp, i32 %and1, i32 %x
  ret i32 %and1.x
; CHECK-LABEL: @test3(
; CHECK: %[[and:.*]] = and i32 %x, -2
; CHECK: ret i32 %[[and]]
}

; CHECK-LABEL: @select_icmp_and_8_eq_0_or_8(
; CHECK-NEXT: [[OR:%[a-z0-9]+]] = or i32 %x, 8
; CHECK-NEXT: ret i32 [[OR]]
define i32 @select_icmp_and_8_eq_0_or_8(i32 %x) {
  %and = and i32 %x, 8
  %cmp = icmp eq i32 %and, 0
  %or = or i32 %x, 8
  %or.x = select i1 %cmp, i32 %or, i32 %x
  ret i32 %or.x
}

; CHECK-LABEL: @select_icmp_and_8_ne_0_and_not_8(
; CHECK-NEXT: [[AND:%[a-z0-9]+]] = and i32 %x, -9
; CHECK-NEXT: ret i32 [[AND]]
define i32 @select_icmp_and_8_ne_0_and_not_8(i32 %x) {
  %and = and i32 %x, 8
  %cmp = icmp eq i32 %and, 0
  %and1 = and i32 %x, -9
  %x.and1 = select i1 %cmp, i32 %x, i32 %and1
  ret i32 %x.and1
}

; CHECK-LABEL: @select_icmp_and_8_eq_0_and_not_8(
; CHECK-NEXT: ret i32 %x
define i32 @select_icmp_and_8_eq_0_and_not_8(i32 %x) {
  %and = and i32 %x, 8
  %cmp = icmp eq i32 %and, 0
  %and1 = and i32 %x, -9
  %and1.x = select i1 %cmp, i32 %and1, i32 %x
  ret i32 %and1.x
}

; CHECK-LABEL: @select_icmp_x_and_8_eq_0_y_and_not_8(
; CHECK: select i1 %cmp, i64 %y, i64 %and1
define i64 @select_icmp_x_and_8_eq_0_y_and_not_8(i32 %x, i64 %y) {
  %and = and i32 %x, 8
  %cmp = icmp eq i32 %and, 0
  %and1 = and i64 %y, -9
  %y.and1 = select i1 %cmp, i64 %y, i64 %and1
  ret i64 %y.and1
}

; CHECK-LABEL: @select_icmp_x_and_8_ne_0_y_and_not_8(
; CHECK: select i1 %cmp, i64 %and1, i64 %y
define i64 @select_icmp_x_and_8_ne_0_y_and_not_8(i32 %x, i64 %y) {
  %and = and i32 %x, 8
  %cmp = icmp eq i32 %and, 0
  %and1 = and i64 %y, -9
  %and1.y = select i1 %cmp, i64 %and1, i64 %y
  ret i64 %and1.y
}


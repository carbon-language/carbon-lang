; RUN: opt < %s -instsimplify -S | FileCheck %s

define i1 @max1(i32 %x, i32 %y) {
; CHECK: @max1
  %c = icmp sgt i32 %x, %y
  %m = select i1 %c, i32 %x, i32 %y
  %r = icmp slt i32 %m, %x
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @max2(i32 %x, i32 %y) {
; CHECK: @max2
  %c = icmp sge i32 %x, %y
  %m = select i1 %c, i32 %x, i32 %y
  %r = icmp sge i32 %m, %x
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @max3(i32 %x, i32 %y) {
; CHECK: @max3
  %c = icmp ugt i32 %x, %y
  %m = select i1 %c, i32 %x, i32 %y
  %r = icmp ult i32 %m, %x
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @max4(i32 %x, i32 %y) {
; CHECK: @max4
  %c = icmp uge i32 %x, %y
  %m = select i1 %c, i32 %x, i32 %y
  %r = icmp uge i32 %m, %x
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @max5(i32 %x, i32 %y) {
; CHECK: @max5
  %c = icmp sgt i32 %x, %y
  %m = select i1 %c, i32 %x, i32 %y
  %r = icmp sgt i32 %x, %m
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @max6(i32 %x, i32 %y) {
; CHECK: @max6
  %c = icmp sge i32 %x, %y
  %m = select i1 %c, i32 %x, i32 %y
  %r = icmp sle i32 %x, %m
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @max7(i32 %x, i32 %y) {
; CHECK: @max7
  %c = icmp ugt i32 %x, %y
  %m = select i1 %c, i32 %x, i32 %y
  %r = icmp ugt i32 %x, %m
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @max8(i32 %x, i32 %y) {
; CHECK: @max8
  %c = icmp uge i32 %x, %y
  %m = select i1 %c, i32 %x, i32 %y
  %r = icmp ule i32 %x, %m
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @min1(i32 %x, i32 %y) {
; CHECK: @min1
  %c = icmp sgt i32 %x, %y
  %m = select i1 %c, i32 %y, i32 %x
  %r = icmp sgt i32 %m, %x
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @min2(i32 %x, i32 %y) {
; CHECK: @min2
  %c = icmp sge i32 %x, %y
  %m = select i1 %c, i32 %y, i32 %x
  %r = icmp sle i32 %m, %x
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @min3(i32 %x, i32 %y) {
; CHECK: @min3
  %c = icmp ugt i32 %x, %y
  %m = select i1 %c, i32 %y, i32 %x
  %r = icmp ugt i32 %m, %x
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @min4(i32 %x, i32 %y) {
; CHECK: @min4
  %c = icmp uge i32 %x, %y
  %m = select i1 %c, i32 %y, i32 %x
  %r = icmp ule i32 %m, %x
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @min5(i32 %x, i32 %y) {
; CHECK: @min5
  %c = icmp sgt i32 %x, %y
  %m = select i1 %c, i32 %y, i32 %x
  %r = icmp slt i32 %x, %m
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @min6(i32 %x, i32 %y) {
; CHECK: @min6
  %c = icmp sge i32 %x, %y
  %m = select i1 %c, i32 %y, i32 %x
  %r = icmp sge i32 %x, %m
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @min7(i32 %x, i32 %y) {
; CHECK: @min7
  %c = icmp ugt i32 %x, %y
  %m = select i1 %c, i32 %y, i32 %x
  %r = icmp ult i32 %x, %m
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @min8(i32 %x, i32 %y) {
; CHECK: @min8
  %c = icmp uge i32 %x, %y
  %m = select i1 %c, i32 %y, i32 %x
  %r = icmp uge i32 %x, %m
  ret i1 %r
; CHECK: ret i1 true
}

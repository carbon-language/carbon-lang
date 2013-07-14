; RUN: opt < %s -instsimplify -S | FileCheck %s

define i1 @max1(i32 %x, i32 %y) {
; CHECK-LABEL: @max1(
  %c = icmp sgt i32 %x, %y
  %m = select i1 %c, i32 %x, i32 %y
  %r = icmp slt i32 %m, %x
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @max2(i32 %x, i32 %y) {
; CHECK-LABEL: @max2(
  %c = icmp sge i32 %x, %y
  %m = select i1 %c, i32 %x, i32 %y
  %r = icmp sge i32 %m, %x
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @max3(i32 %x, i32 %y) {
; CHECK-LABEL: @max3(
  %c = icmp ugt i32 %x, %y
  %m = select i1 %c, i32 %x, i32 %y
  %r = icmp ult i32 %m, %x
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @max4(i32 %x, i32 %y) {
; CHECK-LABEL: @max4(
  %c = icmp uge i32 %x, %y
  %m = select i1 %c, i32 %x, i32 %y
  %r = icmp uge i32 %m, %x
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @max5(i32 %x, i32 %y) {
; CHECK-LABEL: @max5(
  %c = icmp sgt i32 %x, %y
  %m = select i1 %c, i32 %x, i32 %y
  %r = icmp sgt i32 %x, %m
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @max6(i32 %x, i32 %y) {
; CHECK-LABEL: @max6(
  %c = icmp sge i32 %x, %y
  %m = select i1 %c, i32 %x, i32 %y
  %r = icmp sle i32 %x, %m
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @max7(i32 %x, i32 %y) {
; CHECK-LABEL: @max7(
  %c = icmp ugt i32 %x, %y
  %m = select i1 %c, i32 %x, i32 %y
  %r = icmp ugt i32 %x, %m
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @max8(i32 %x, i32 %y) {
; CHECK-LABEL: @max8(
  %c = icmp uge i32 %x, %y
  %m = select i1 %c, i32 %x, i32 %y
  %r = icmp ule i32 %x, %m
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @min1(i32 %x, i32 %y) {
; CHECK-LABEL: @min1(
  %c = icmp sgt i32 %x, %y
  %m = select i1 %c, i32 %y, i32 %x
  %r = icmp sgt i32 %m, %x
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @min2(i32 %x, i32 %y) {
; CHECK-LABEL: @min2(
  %c = icmp sge i32 %x, %y
  %m = select i1 %c, i32 %y, i32 %x
  %r = icmp sle i32 %m, %x
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @min3(i32 %x, i32 %y) {
; CHECK-LABEL: @min3(
  %c = icmp ugt i32 %x, %y
  %m = select i1 %c, i32 %y, i32 %x
  %r = icmp ugt i32 %m, %x
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @min4(i32 %x, i32 %y) {
; CHECK-LABEL: @min4(
  %c = icmp uge i32 %x, %y
  %m = select i1 %c, i32 %y, i32 %x
  %r = icmp ule i32 %m, %x
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @min5(i32 %x, i32 %y) {
; CHECK-LABEL: @min5(
  %c = icmp sgt i32 %x, %y
  %m = select i1 %c, i32 %y, i32 %x
  %r = icmp slt i32 %x, %m
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @min6(i32 %x, i32 %y) {
; CHECK-LABEL: @min6(
  %c = icmp sge i32 %x, %y
  %m = select i1 %c, i32 %y, i32 %x
  %r = icmp sge i32 %x, %m
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @min7(i32 %x, i32 %y) {
; CHECK-LABEL: @min7(
  %c = icmp ugt i32 %x, %y
  %m = select i1 %c, i32 %y, i32 %x
  %r = icmp ult i32 %x, %m
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @min8(i32 %x, i32 %y) {
; CHECK-LABEL: @min8(
  %c = icmp uge i32 %x, %y
  %m = select i1 %c, i32 %y, i32 %x
  %r = icmp uge i32 %x, %m
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @maxmin1(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: @maxmin1(
  %c1 = icmp sge i32 %x, %y
  %max = select i1 %c1, i32 %x, i32 %y
  %c2 = icmp sge i32 %x, %z
  %min = select i1 %c2, i32 %z, i32 %x
  %c = icmp sge i32 %max, %min
  ret i1 %c
; CHECK: ret i1 true
}

define i1 @maxmin2(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: @maxmin2(
  %c1 = icmp sge i32 %x, %y
  %max = select i1 %c1, i32 %x, i32 %y
  %c2 = icmp sge i32 %x, %z
  %min = select i1 %c2, i32 %z, i32 %x
  %c = icmp sgt i32 %min, %max
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @maxmin3(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: @maxmin3(
  %c1 = icmp sge i32 %x, %y
  %max = select i1 %c1, i32 %x, i32 %y
  %c2 = icmp sge i32 %x, %z
  %min = select i1 %c2, i32 %z, i32 %x
  %c = icmp sle i32 %min, %max
  ret i1 %c
; CHECK: ret i1 true
}

define i1 @maxmin4(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: @maxmin4(
  %c1 = icmp sge i32 %x, %y
  %max = select i1 %c1, i32 %x, i32 %y
  %c2 = icmp sge i32 %x, %z
  %min = select i1 %c2, i32 %z, i32 %x
  %c = icmp slt i32 %max, %min
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @maxmin5(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: @maxmin5(
  %c1 = icmp uge i32 %x, %y
  %max = select i1 %c1, i32 %x, i32 %y
  %c2 = icmp uge i32 %x, %z
  %min = select i1 %c2, i32 %z, i32 %x
  %c = icmp uge i32 %max, %min
  ret i1 %c
; CHECK: ret i1 true
}

define i1 @maxmin6(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: @maxmin6(
  %c1 = icmp uge i32 %x, %y
  %max = select i1 %c1, i32 %x, i32 %y
  %c2 = icmp uge i32 %x, %z
  %min = select i1 %c2, i32 %z, i32 %x
  %c = icmp ugt i32 %min, %max
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @maxmin7(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: @maxmin7(
  %c1 = icmp uge i32 %x, %y
  %max = select i1 %c1, i32 %x, i32 %y
  %c2 = icmp uge i32 %x, %z
  %min = select i1 %c2, i32 %z, i32 %x
  %c = icmp ule i32 %min, %max
  ret i1 %c
; CHECK: ret i1 true
}

define i1 @maxmin8(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: @maxmin8(
  %c1 = icmp uge i32 %x, %y
  %max = select i1 %c1, i32 %x, i32 %y
  %c2 = icmp uge i32 %x, %z
  %min = select i1 %c2, i32 %z, i32 %x
  %c = icmp ult i32 %max, %min
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @eqcmp1(i32 %x, i32 %y) {
; CHECK-LABEL: @eqcmp1(
  %c = icmp sge i32 %x, %y
  %max = select i1 %c, i32 %x, i32 %y
  %r = icmp eq i32 %max, %x
  ret i1 %r
; CHECK: ret i1 %c
}

define i1 @eqcmp2(i32 %x, i32 %y) {
; CHECK-LABEL: @eqcmp2(
  %c = icmp sge i32 %x, %y
  %max = select i1 %c, i32 %x, i32 %y
  %r = icmp eq i32 %x, %max
  ret i1 %r
; CHECK: ret i1 %c
}

define i1 @eqcmp3(i32 %x, i32 %y) {
; CHECK-LABEL: @eqcmp3(
  %c = icmp uge i32 %x, %y
  %max = select i1 %c, i32 %x, i32 %y
  %r = icmp eq i32 %max, %x
  ret i1 %r
; CHECK: ret i1 %c
}

define i1 @eqcmp4(i32 %x, i32 %y) {
; CHECK-LABEL: @eqcmp4(
  %c = icmp uge i32 %x, %y
  %max = select i1 %c, i32 %x, i32 %y
  %r = icmp eq i32 %x, %max
  ret i1 %r
; CHECK: ret i1 %c
}

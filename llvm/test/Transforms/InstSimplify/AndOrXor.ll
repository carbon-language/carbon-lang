; RUN: opt < %s -instsimplify -S | FileCheck %s

define i64 @pow2(i32 %x) {
; CHECK-LABEL: @pow2(
  %negx = sub i32 0, %x
  %x2 = and i32 %x, %negx
  %e = zext i32 %x2 to i64
  %nege = sub i64 0, %e
  %e2 = and i64 %e, %nege
  ret i64 %e2
; CHECK: ret i64 %e
}

define i64 @pow2b(i32 %x) {
; CHECK-LABEL: @pow2b(
  %sh = shl i32 2, %x
  %e = zext i32 %sh to i64
  %nege = sub i64 0, %e
  %e2 = and i64 %e, %nege
  ret i64 %e2
; CHECK: ret i64 %e
}

define i32 @sub_neg_nuw(i32 %x, i32 %y) {
; CHECK-LABEL: @sub_neg_nuw(
  %neg = sub nuw i32 0, %y
  %sub = sub i32 %x, %neg
  ret i32 %sub
; CHECK: ret i32 %x
}

define i1 @and_of_icmps0(i32 %b) {
; CHECK-LABEL: @and_of_icmps0(
  %1 = add i32 %b, 2
  %2 = icmp ult i32 %1, 4
  %cmp3 = icmp sgt i32 %b, 2
  %cmp = and i1 %2, %cmp3
  ret i1 %cmp
; CHECK: ret i1 false
}

define i1 @and_of_icmps1(i32 %b) {
; CHECK-LABEL: @and_of_icmps1(
  %1 = add nsw i32 %b, 2
  %2 = icmp slt i32 %1, 4
  %cmp3 = icmp sgt i32 %b, 2
  %cmp = and i1 %2, %cmp3
  ret i1 %cmp
; CHECK: ret i1 false
}

define i1 @and_of_icmps2(i32 %b) {
; CHECK-LABEL: @and_of_icmps2(
  %1 = add i32 %b, 2
  %2 = icmp ule i32 %1, 3
  %cmp3 = icmp sgt i32 %b, 2
  %cmp = and i1 %2, %cmp3
  ret i1 %cmp
; CHECK: ret i1 false
}

define i1 @and_of_icmps3(i32 %b) {
; CHECK-LABEL: @and_of_icmps3(
  %1 = add nsw i32 %b, 2
  %2 = icmp sle i32 %1, 3
  %cmp3 = icmp sgt i32 %b, 2
  %cmp = and i1 %2, %cmp3
  ret i1 %cmp
; CHECK: ret i1 false
}

define i1 @and_of_icmps4(i32 %b) {
; CHECK-LABEL: @and_of_icmps4(
  %1 = add nuw i32 %b, 2
  %2 = icmp ult i32 %1, 4
  %cmp3 = icmp ugt i32 %b, 2
  %cmp = and i1 %2, %cmp3
  ret i1 %cmp
; CHECK: ret i1 false
}

define i1 @and_of_icmps5(i32 %b) {
; CHECK-LABEL: @and_of_icmps5(
  %1 = add nuw i32 %b, 2
  %2 = icmp ule i32 %1, 3
  %cmp3 = icmp ugt i32 %b, 2
  %cmp = and i1 %2, %cmp3
  ret i1 %cmp
; CHECK: ret i1 false
}

define i1 @or_of_icmps0(i32 %b) {
; CHECK-LABEL: @or_of_icmps0(
  %1 = add i32 %b, 2
  %2 = icmp uge i32 %1, 4
  %cmp3 = icmp sle i32 %b, 2
  %cmp = or i1 %2, %cmp3
  ret i1 %cmp
; CHECK: ret i1 true
}

define i1 @or_of_icmps1(i32 %b) {
; CHECK-LABEL: @or_of_icmps1(
  %1 = add nsw i32 %b, 2
  %2 = icmp sge i32 %1, 4
  %cmp3 = icmp sle i32 %b, 2
  %cmp = or i1 %2, %cmp3
  ret i1 %cmp
; CHECK: ret i1 true
}

define i1 @or_of_icmps2(i32 %b) {
; CHECK-LABEL: @or_of_icmps2(
  %1 = add i32 %b, 2
  %2 = icmp ugt i32 %1, 3
  %cmp3 = icmp sle i32 %b, 2
  %cmp = or i1 %2, %cmp3
  ret i1 %cmp
; CHECK: ret i1 true
}

define i1 @or_of_icmps3(i32 %b) {
; CHECK-LABEL: @or_of_icmps3(
  %1 = add nsw i32 %b, 2
  %2 = icmp sgt i32 %1, 3
  %cmp3 = icmp sle i32 %b, 2
  %cmp = or i1 %2, %cmp3
  ret i1 %cmp
; CHECK: ret i1 true
}

define i1 @or_of_icmps4(i32 %b) {
; CHECK-LABEL: @or_of_icmps4(
  %1 = add nuw i32 %b, 2
  %2 = icmp uge i32 %1, 4
  %cmp3 = icmp ule i32 %b, 2
  %cmp = or i1 %2, %cmp3
  ret i1 %cmp
; CHECK: ret i1 true
}

define i1 @or_of_icmps5(i32 %b) {
; CHECK-LABEL: @or_of_icmps5(
  %1 = add nuw i32 %b, 2
  %2 = icmp ugt i32 %1, 3
  %cmp3 = icmp ule i32 %b, 2
  %cmp = or i1 %2, %cmp3
  ret i1 %cmp
; CHECK: ret i1 true
}

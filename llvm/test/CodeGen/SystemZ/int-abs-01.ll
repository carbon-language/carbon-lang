; Test integer absolute.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test i32->i32 absolute using slt.
define i32 @f1(i32 %val) {
; CHECK-LABEL: f1:
; CHECK: lpr %r2, %r2
; CHECK: br %r14
  %cmp = icmp slt i32 %val, 0
  %neg = sub i32 0, %val
  %res = select i1 %cmp, i32 %neg, i32 %val
  ret i32 %res
}

; Test i32->i32 absolute using sle.
define i32 @f2(i32 %val) {
; CHECK-LABEL: f2:
; CHECK: lpr %r2, %r2
; CHECK: br %r14
  %cmp = icmp sle i32 %val, 0
  %neg = sub i32 0, %val
  %res = select i1 %cmp, i32 %neg, i32 %val
  ret i32 %res
}

; Test i32->i32 absolute using sgt.
define i32 @f3(i32 %val) {
; CHECK-LABEL: f3:
; CHECK: lpr %r2, %r2
; CHECK: br %r14
  %cmp = icmp sgt i32 %val, 0
  %neg = sub i32 0, %val
  %res = select i1 %cmp, i32 %val, i32 %neg
  ret i32 %res
}

; Test i32->i32 absolute using sge.
define i32 @f4(i32 %val) {
; CHECK-LABEL: f4:
; CHECK: lpr %r2, %r2
; CHECK: br %r14
  %cmp = icmp sge i32 %val, 0
  %neg = sub i32 0, %val
  %res = select i1 %cmp, i32 %val, i32 %neg
  ret i32 %res
}

; Test i32->i64 absolute.
define i64 @f5(i32 %val) {
; CHECK-LABEL: f5:
; CHECK: lpgfr %r2, %r2
; CHECK: br %r14
  %ext = sext i32 %val to i64
  %cmp = icmp slt i64 %ext, 0
  %neg = sub i64 0, %ext
  %res = select i1 %cmp, i64 %neg, i64 %ext
  ret i64 %res
}

; Test i32->i64 absolute that uses an "in-register" form of sign extension.
define i64 @f6(i64 %val) {
; CHECK-LABEL: f6:
; CHECK: lpgfr %r2, %r2
; CHECK: br %r14
  %trunc = trunc i64 %val to i32
  %ext = sext i32 %trunc to i64
  %cmp = icmp slt i64 %ext, 0
  %neg = sub i64 0, %ext
  %res = select i1 %cmp, i64 %neg, i64 %ext
  ret i64 %res
}

; Test i64 absolute.
define i64 @f7(i64 %val) {
; CHECK-LABEL: f7:
; CHECK: lpgr %r2, %r2
; CHECK: br %r14
  %cmp = icmp slt i64 %val, 0
  %neg = sub i64 0, %val
  %res = select i1 %cmp, i64 %neg, i64 %val
  ret i64 %res
}

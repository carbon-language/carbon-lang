; RUN: opt < %s -nary-reassociate -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

declare void @foo(float*)

; foo(&a[i]);
; foo(&a[i + j]);
;   =>
; t = &a[i];
; foo(t);
; foo(t + j);
define void @reassociate_gep(float* %a, i64 %i, i64 %j) {
; CHECK-LABEL: @reassociate_gep(
  %1 = add i64 %i, %j
  %2 = getelementptr float, float* %a, i64 %i
; CHECK: [[t1:[^ ]+]] = getelementptr float, float* %a, i64 %i
  call void @foo(float* %2)
; CHECK: call void @foo(float* [[t1]])
  %3 = getelementptr float, float* %a, i64 %1
; CHECK: [[t2:[^ ]+]] = getelementptr float, float* [[t1]], i64 %j
  call void @foo(float* %3)
; CHECK: call void @foo(float* [[t2]])
  ret void
}

; foo(&a[sext(j)]);
; foo(&a[sext(i +nsw j)]);
;   =>
; t = &a[sext(j)];
; foo(t);
; foo(t + sext(i));
define void @reassociate_gep_nsw(float* %a, i32 %i, i32 %j) {
; CHECK-LABEL: @reassociate_gep_nsw(
  %1 = add nsw i32 %i, %j
  %idxprom.1 = sext i32 %1 to i64
  %idxprom.j = sext i32 %j to i64
  %2 = getelementptr float, float* %a, i64 %idxprom.j
; CHECK: [[t1:[^ ]+]] = getelementptr float, float* %a, i64 %idxprom.j
  call void @foo(float* %2)
; CHECK: call void @foo(float* [[t1]])
  %3 = getelementptr float, float* %a, i64 %idxprom.1
; CHECK: [[sexti:[^ ]+]] = sext i32 %i to i64
; CHECK: [[t2:[^ ]+]] = getelementptr float, float* [[t1]], i64 [[sexti]]
  call void @foo(float* %3)
; CHECK: call void @foo(float* [[t2]])
  ret void
}

; Do not split the second GEP because sext(i + j) != sext(i) + sext(j).
define void @reassociate_gep_no_nsw(float* %a, i32 %i, i32 %j) {
; CHECK-LABEL: @reassociate_gep_no_nsw(
  %1 = add i32 %i, %j
  %2 = getelementptr float, float* %a, i32 %j
; CHECK: getelementptr float, float* %a, i32 %j
  call void @foo(float* %2)
  %3 = getelementptr float, float* %a, i32 %1
; CHECK: getelementptr float, float* %a, i32 %1
  call void @foo(float* %3)
  ret void
}

define void @reassociate_gep_128(float* %a, i128 %i, i128 %j) {
; CHECK-LABEL: @reassociate_gep_128(
  %1 = add i128 %i, %j
  %2 = getelementptr float, float* %a, i128 %i
; CHECK: [[t1:[^ ]+]] = getelementptr float, float* %a, i128 %i
  call void @foo(float* %2)
; CHECK: call void @foo(float* [[t1]])
  %3 = getelementptr float, float* %a, i128 %1
; CHECK: [[truncj:[^ ]+]] = trunc i128 %j to i64
; CHECK: [[t2:[^ ]+]] = getelementptr float, float* [[t1]], i64 [[truncj]]
  call void @foo(float* %3)
; CHECK: call void @foo(float* [[t2]])
  ret void
}

; RUN: opt -instcombine -S -o - %s | FileCheck %s
; Tests that we preserve the inrange attribute on indices where possible.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { i32 (...)** }

@vt = external global [3 x i8*]

; CHECK: define i32 (...)** @f0()
define i32 (...)** @f0() {
  ; CHECK-NEXT: ret i32 (...)** bitcast (i8** getelementptr inbounds ([3 x i8*], [3 x i8*]* @vt, inrange i64 0, i64 2) to i32 (...)**
  ret i32 (...)** getelementptr (i32 (...)*, i32 (...)** bitcast (i8** getelementptr inbounds ([3 x i8*], [3 x i8*]* @vt, inrange i64 0, i64 1) to i32 (...)**), i64 1)
}

; CHECK: define i32 (...)** @f1()
define i32 (...)** @f1() {
  ; CHECK-NEXT: ret i32 (...)** getelementptr (i32 (...)*, i32 (...)** bitcast (i8** getelementptr inbounds ([3 x i8*], [3 x i8*]* @vt, i64 0, inrange i64 1) to i32 (...)**), i64 1)
  ret i32 (...)** getelementptr (i32 (...)*, i32 (...)** bitcast (i8** getelementptr inbounds ([3 x i8*], [3 x i8*]* @vt, i64 0, inrange i64 1) to i32 (...)**), i64 1)
}

; CHECK: define i32 (...)** @f2()
define i32 (...)** @f2() {
  ; CHECK-NEXT: ret i32 (...)** getelementptr (i32 (...)*, i32 (...)** bitcast (i8** getelementptr inbounds ([3 x i8*], [3 x i8*]* @vt, i64 0, inrange i64 1) to i32 (...)**), i64 3)
  ret i32 (...)** getelementptr (i32 (...)*, i32 (...)** bitcast (i8** getelementptr inbounds ([3 x i8*], [3 x i8*]* @vt, i64 0, inrange i64 1) to i32 (...)**), i64 3)
}

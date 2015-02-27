; RUN: opt < %s -analyze -basicaa -da -da-delinearize=false | FileCheck %s
; RUN: opt < %s -analyze -basicaa -da -da-delinearize | FileCheck %s -check-prefix=DELIN

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.6.0"

; for (int i = 0; i < 100; ++i) {
;   int t0 = a[i][i];
;   int t1 = t0 + 1;
;   a[i][5] = t1;
; }
; The subscript 5 in a[i][5] is deliberately an i32, mismatching the types of
; other subscript. DependenceAnalysis before the fix crashed due to this
; mismatch.
define void @i32_subscript([100 x [100 x i32]]* %a, i32* %b) {
; CHECK-LABEL: 'Dependence Analysis' for function 'i32_subscript'
; DELIN-LABEL: 'Dependence Analysis' for function 'i32_subscript'
entry:
  br label %for.body

for.body:
; CHECK: da analyze - none!
; CHECK: da analyze - anti [=|<]!
; CHECK: da analyze - none!
; DELIN: da analyze - none!
; DELIN: da analyze - anti [=|<]!
; DELIN: da analyze - none!
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.body ]
  %a.addr = getelementptr [100 x [100 x i32]], [100 x [100 x i32]]* %a, i64 0, i64 %i, i64 %i
  %a.addr.2 = getelementptr [100 x [100 x i32]], [100 x [100 x i32]]* %a, i64 0, i64 %i, i32 5
  %0 = load i32, i32* %a.addr, align 4
  %1 = add i32 %0, 1
  store i32 %1, i32* %a.addr.2, align 4
  %i.inc = add nsw i64 %i, 1
  %exitcond = icmp ne i64 %i.inc, 100
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

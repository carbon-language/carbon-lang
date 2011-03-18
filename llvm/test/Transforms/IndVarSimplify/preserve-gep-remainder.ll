; RUN: opt < %s -indvars -S \
; RUN:   | grep {\[%\]p.2.ip.1 = getelementptr \\\[3 x \\\[3 x double\\\]\\\]\\* \[%\]p, i64 2, i64 \[%\]tmp, i64 1}
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128-n:32:64"

; Indvars shouldn't expand this to
;   %p.2.ip.1 = getelementptr [3 x [3 x double]]* %p, i64 0, i64 %tmp, i64 19
; or something. That's valid, but more obscure.

define void @foo([3 x [3 x double]]* noalias %p) nounwind {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %ip = add i64 %i, 1
  %p.2.ip.1 = getelementptr [3 x [3 x double]]* %p, i64 2, i64 %ip, i64 1
  volatile store double 0.0, double* %p.2.ip.1
  %i.next = add i64 %i, 1
  br label %loop
}

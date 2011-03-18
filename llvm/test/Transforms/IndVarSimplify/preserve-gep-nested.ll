; RUN: opt < %s -indvars -S > %t
; Exactly one getelementptr for each load+store.
; RUN: grep getelementptr %t | count 6
; Each getelementptr using %struct.Q* %s as a base and not i8*.
; RUN: grep {getelementptr \[%\]struct\\.Q\\* \[%\]s,} %t | count 6
; No explicit integer multiplications!
; RUN: not grep {= mul} %t
; No i8* arithmetic or pointer casting anywhere!
; RUN: not grep {i8\\*} %t
; RUN: not grep bitcast %t
; RUN: not grep inttoptr %t
; RUN: not grep ptrtoint %t

; FIXME: This test should pass with or without TargetData. Until opt
; supports running tests without targetdata, just hardware this in.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n:32:64"

%struct.Q = type { [10 x %struct.N] }
%struct.N = type { %struct.S }
%struct.S = type { [100 x double], [100 x double] }

define void @foo(%struct.Q* %s, i64 %n) nounwind {
entry:
  br label %bb1

bb1:
  %i = phi i64 [ 2, %entry ], [ %i.next, %bb ]
  %j = phi i64 [ 0, %entry ], [ %j.next, %bb ]
  %t5 = icmp slt i64 %i, %n
  br i1 %t5, label %bb, label %return

bb:
  %t0 = getelementptr inbounds %struct.Q* %s, i64 0, i32 0, i64 0, i32 0, i32 0, i64 %i
  %t1 = load double* %t0, align 8
  %t2 = fmul double %t1, 3.200000e+00
  %t3 = getelementptr inbounds %struct.Q* %s, i64 0, i32 0, i64 0, i32 0, i32 0, i64 %i
  store double %t2, double* %t3, align 8

  %s0 = getelementptr inbounds %struct.Q* %s, i64 13, i32 0, i64 7, i32 0, i32 1, i64 %i
  %s1 = load double* %s0, align 8
  %s2 = fmul double %s1, 3.200000e+00
  %s3 = getelementptr inbounds %struct.Q* %s, i64 13, i32 0, i64 7, i32 0, i32 1, i64 %i
  store double %s2, double* %s3, align 8

  %u0 = getelementptr inbounds %struct.Q* %s, i64 0, i32 0, i64 7, i32 0, i32 1, i64 %j
  %u1 = load double* %u0, align 8
  %u2 = fmul double %u1, 3.200000e+00
  %u3 = getelementptr inbounds %struct.Q* %s, i64 0, i32 0, i64 7, i32 0, i32 1, i64 %j
  store double %u2, double* %u3, align 8

  %v0 = getelementptr inbounds %struct.Q* %s, i64 0, i32 0, i64 0, i32 0, i32 1, i64 %i
  %v1 = load double* %v0, align 8
  %v2 = fmul double %v1, 3.200000e+00
  %v3 = getelementptr inbounds %struct.Q* %s, i64 0, i32 0, i64 0, i32 0, i32 1, i64 %i
  store double %v2, double* %v3, align 8

  %w0 = getelementptr inbounds %struct.Q* %s, i64 0, i32 0, i64 0, i32 0, i32 0, i64 %j
  %w1 = load double* %w0, align 8
  %w2 = fmul double %w1, 3.200000e+00
  %w3 = getelementptr inbounds %struct.Q* %s, i64 0, i32 0, i64 0, i32 0, i32 0, i64 %j
  store double %w2, double* %w3, align 8

  %x0 = getelementptr inbounds %struct.Q* %s, i64 0, i32 0, i64 3, i32 0, i32 0, i64 %i
  %x1 = load double* %x0, align 8
  %x2 = fmul double %x1, 3.200000e+00
  %x3 = getelementptr inbounds %struct.Q* %s, i64 0, i32 0, i64 3, i32 0, i32 0, i64 %i
  store double %x2, double* %x3, align 8

  %i.next = add i64 %i, 1
  %j.next = add i64 %j, 1
  br label %bb1

return:
  ret void
}

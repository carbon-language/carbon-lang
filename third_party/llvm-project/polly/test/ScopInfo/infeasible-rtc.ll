; RUN: opt %loadPolly -polly-print-detect -disable-output < %s \
; RUN:  | FileCheck %s -check-prefix=DETECT

; RUN: opt %loadPolly -polly-print-scops -disable-output < %s \
; RUN:  | FileCheck %s -check-prefix=SCOPS

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; DETECT: Valid Region for Scop: test1.header => test1.exit
; SCOPS-NOT: Region: %test1.header---%test1.exit

; Verify that we detect this scop, but that, due to an infeasible run-time
; check, we refuse to model it.

define void @test(i64* %a) nounwind uwtable {
preheader:
  br label %test1.header

test1.header:
  %i = phi i56 [ 0, %preheader ], [ %i.1, %test1.header ]
  %tmp = zext i56 %i to i64
  %A.addr = getelementptr i64, i64* %a, i64 %tmp
  %A.load = load i64, i64* %A.addr, align 4
  %A.inc = zext i56 %i to i64
  %A.val = add nsw i64 %A.load, %A.inc
  store i64 %A.val, i64* %A.addr, align 4
  %i.1 = add i56 %i, 1
  %exitcond = icmp eq i56 %i.1, 0
  br i1 %exitcond, label %test1.exit, label %test1.header

test1.exit:
  ret void
}

; Old version of the previous test; make sure we compute the trip count
; correctly.

; SCOPS: { Stmt_header[i0] : 0 <= i0 <= 127 };

define void @test2([128 x i32]* %a) nounwind uwtable {
preheader:
  br label %header

header:
  %i = phi i7 [ 0, %preheader ], [ %i.1, %header ]
  %tmp = zext i7 %i to i64
  %A.addr = getelementptr [128 x i32], [128 x i32]* %a, i64 0, i64 %tmp
  %A.load = load i32, i32* %A.addr, align 4
  %A.inc = zext i7 %i to i32
  %A.val = add nsw i32 %A.load, %A.inc
  store i32 %A.val, i32* %A.addr, align 4
  %i.1 = add i7 %i, 1
  %exitcond = icmp eq i7 %i.1, 0
  br i1 %exitcond, label %exit, label %header

exit:
  ret void
}

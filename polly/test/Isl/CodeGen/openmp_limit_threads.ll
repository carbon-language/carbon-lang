; RUN: opt %loadPolly -polly-codegen -enable-polly-openmp -S < %s | FileCheck %s --check-prefix=AUTO
; RUN: opt %loadPolly -polly-codegen -enable-polly-openmp -polly-num-threads=1 -S < %s | FileCheck %s --check-prefix=ONE
; RUN: opt %loadPolly -polly-codegen -enable-polly-openmp -polly-num-threads=4 -S < %s | FileCheck %s --check-prefix=FOUR
;
; AUTO: call void @GOMP_parallel_loop_runtime_start(void (i8*)* @jd.polly.subfn, i8* %polly.par.userContext{{[0-9]*}}, i64 0, i64 0, i64 1024, i64 1)
; ONE: call void @GOMP_parallel_loop_runtime_start(void (i8*)* @jd.polly.subfn, i8* %polly.par.userContext{{[0-9]*}}, i64 1, i64 0, i64 1024, i64 1)
; FOUR: call void @GOMP_parallel_loop_runtime_start(void (i8*)* @jd.polly.subfn, i8* %polly.par.userContext{{[0-9]*}}, i64 4, i64 0, i64 1024, i64 1)
;
;    void jd(int *A) {
;      for (int i = 0; i < 1024; i++)
;        A[i] = 0;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32* %A, i64 %indvars.iv
  store i32 0, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

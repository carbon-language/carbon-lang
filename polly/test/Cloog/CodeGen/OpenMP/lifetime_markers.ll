; RUN: opt %loadPolly -S -polly-codegen -enable-polly-openmp < %s | FileCheck %s
;
; Check that we allocate the parallel context in the entry block and use
; lifetime markers to mark the live range.
;
; CHECK: entry:
; CHECK:   %polly.par.userContext = alloca { i32* }
; CHECK:   br label %while.cond
;
; CHECK:       polly.start:
; CHECK-NEXT:    %[[BC1:[._0-9a-zA-Z]*]] = bitcast { i32* }* %polly.par.userContext to i8*
; CHECK-NEXT:    call void @llvm.lifetime.start(i64 8, i8* %[[BC1]])
; CHECK-NEXT:    %[[GEP:[._0-9a-zA-Z]*]] = getelementptr inbounds { i32* }* %polly.par.userContext, i32 0, i32 0
; CHECK-NEXT:    store i32* %A, i32** %[[GEP]]
; CHECK-NEXT:    %polly.par.userContext{{[0-9]*}} = bitcast { i32* }* %polly.par.userContext to i8*
; CHECK-NEXT:    call void @GOMP_parallel_loop_runtime_start(void (i8*)* @jd.polly.subfn, i8* %polly.par.userContext{{[0-9]*}}, i64 0, i64 0, i64 1024, i64 1)
; CHECK-NEXT:    call void @jd.polly.subfn(i8* %polly.par.userContext{{[0-9]*}})
; CHECK-NEXT:    call void @GOMP_parallel_end()
; CHECK-NEXT:    %[[BC2:[._0-9a-zA-Z]*]] = bitcast { i32* }* %polly.par.userContext to i8*
; CHECK-NEXT:    call void @llvm.lifetime.end(i64 8, i8* %[[BC2]])
; CHECK-NEXT:    br label %polly.merge_new_and_old

;    int cond();
;    void jd(int *A) {
;      while (cond())
;        for (int j = 0; j < 1024; j++)
;          A[j] = 1;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* %A) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %for.end, %entry
  %call = call i32 (...)* @cond() #2
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %while.end, label %while.body

while.body:                                       ; preds = %while.cond
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %while.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %while.body ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32* %A, i64 %indvars.iv
  store i32 1, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %while.cond

while.end:                                        ; preds = %while.cond
  ret void
}

declare i32 @cond(...) #1

; RUN: opt %loadPolly -polly-ast -analyze  -S < %s | FileCheck %s

;#include "limits.h"
;#define N 20
;
;int main () {
;  long long i;
;  long long A[N];
;
;  A[0] = 0;
;
;  __sync_synchronize();
;
;  for (i = 0; i < LLONG_MAX; i++)
;    A[0] = i;
;
;  __sync_synchronize();
;
;  if (A[0] == LLONG_MAX - 1)
;    return 0;
;  else
;    return 1;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define i32 @main() nounwind {
entry:
  %A = alloca [20 x i64], align 8                 ; <[20 x i64]*> [#uses=3]
  %arraydecay = getelementptr inbounds [20 x i64], [20 x i64]* %A, i32 0, i32 0 ; <i64*> [#uses=1]
  %arrayidx = getelementptr inbounds i64, i64* %arraydecay, i64 0 ; <i64*> [#uses=1]
  store i64 0, i64* %arrayidx
  fence seq_cst
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]  ; <i64> [#uses=3]
  %exitcond = icmp ne i64 %0, 9223372036854775807 ; <i1> [#uses=1]
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arraydecay2 = getelementptr inbounds [20 x i64], [20 x i64]* %A, i32 0, i32 0 ; <i64*> [#uses=1]
  %arrayidx3 = getelementptr inbounds i64, i64* %arraydecay2, i64 0 ; <i64*> [#uses=1]
  store i64 %0, i64* %arrayidx3
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i64 %0, 1                        ; <i64> [#uses=1]
  br label %for.cond

for.end:                                          ; preds = %for.cond
  fence seq_cst
  %arraydecay5 = getelementptr inbounds [20 x i64], [20 x i64]* %A, i32 0, i32 0 ; <i64*> [#uses=1]
  %arrayidx6 = getelementptr inbounds i64, i64* %arraydecay5, i64 0 ; <i64*> [#uses=1]
  %tmp7 = load i64, i64* %arrayidx6                    ; <i64> [#uses=1]
  %cmp8 = icmp eq i64 %tmp7, 9223372036854775806  ; <i1> [#uses=1]
  br i1 %cmp8, label %if.then, label %if.else

if.then:                                          ; preds = %for.end
  br label %return

if.else:                                          ; preds = %for.end
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ 0, %if.then ], [ 1, %if.else ] ; <i32> [#uses=1]
  ret i32 %retval.0
}

; CHECK: for (int c0 = 0; c0 <= 9223372036854775806; c0 += 1)
; CHECK:   Stmt_for_body(c0);

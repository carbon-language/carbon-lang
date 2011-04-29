; RUN: opt %loadPolly %defaultOpts -polly-cloog -analyze  -S < %s | FileCheck %s
; RUN: opt %loadPolly %defaultOpts -polly-codegen -O3 < %s
; ModuleID = 'single_do_loop_ll_max_iterations.s'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [20 x i64] zeroinitializer, align 8 ; <[20 x i64]*> [#uses=1]

define i32 @main() nounwind {
entry:
  volatile store i64 0, i64* getelementptr inbounds ([20 x i64]* @A, i32 0, i32 0)
  call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 false)
  br label %do.body

do.body:                                          ; preds = %do.cond, %entry
  %0 = phi i64 [ 0, %entry ], [ %inc, %do.cond ]  ; <i64> [#uses=2]
  volatile store i64 %0, i64* getelementptr inbounds ([20 x i64]* @A, i32 0, i32 0)
  %inc = add nsw i64 %0, 1                        ; <i64> [#uses=2]
  br label %do.cond

do.cond:                                          ; preds = %do.body
  %exitcond = icmp ne i64 %inc, 9223372036854775807 ; <i1> [#uses=1]
  br i1 %exitcond, label %do.body, label %do.end

do.end:                                           ; preds = %do.cond
  call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 false)
  %tmp3 = volatile load i64* getelementptr inbounds ([20 x i64]* @A, i32 0, i32 0) ; <i64> [#uses=1]
  %cmp4 = icmp eq i64 %tmp3, 9223372036854775806  ; <i1> [#uses=1]
  br i1 %cmp4, label %if.then, label %if.else

if.then:                                          ; preds = %do.end
  br label %return

if.else:                                          ; preds = %do.end
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ 0, %if.then ], [ 1, %if.else ] ; <i32> [#uses=1]
  ret i32 %retval.0
}

declare void @llvm.memory.barrier(i1, i1, i1, i1, i1) nounwind
; CHECK:for (c2=0;c2<=9223372036854775806;c2++) {


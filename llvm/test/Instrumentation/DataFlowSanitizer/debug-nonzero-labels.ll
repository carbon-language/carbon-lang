; RUN: opt < %s -dfsan -dfsan-args-abi -dfsan-debug-nonzero-labels -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare i32 @g()

; CHECK: define { i32, i16 } @"dfs$f"(i32, i16)
define i32 @f(i32) {
  ; CHECK: [[LOCALLABELALLOCA:%.*]] = alloca i16
  ; CHECK: [[ARGCMP:%.*]] = icmp ne i16 %1, 0
  ; CHECK: br i1 [[ARGCMP]]
  %i = alloca i32
  store i32 %0, i32* %i
  ; CHECK: [[CALL:%.*]] = call { i32, i16 } @"dfs$g"()
  ; CHECK: [[CALLLABEL:%.*]] = extractvalue { i32, i16 } [[CALL]], 1
  ; CHECK: [[CALLCMP:%.*]] = icmp ne i16 [[CALLLABEL]], 0
  ; CHECK: br i1 [[CALLCMP]]
  %call = call i32 @g()
  ; CHECK: [[LOCALLABEL:%.*]] = load i16* [[LOCALLABELALLOCA]]
  ; CHECK: [[LOCALCMP:%.*]] = icmp ne i16 [[LOCALLABEL]], 0
  ; CHECK: br i1 [[LOCALCMP]]
  %load = load i32* %i
  ret i32 %load
}

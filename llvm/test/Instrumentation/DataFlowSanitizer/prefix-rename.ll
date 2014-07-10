; RUN: opt < %s -dfsan -S | FileCheck %s
; RUN: opt < %s -dfsan -dfsan-args-abi -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; CHECK: module asm ".symver dfs$f1,dfs$f@@version1"
module asm ".symver f1,f@@version1"

; CHECK: @"dfs$f2" = alias {{.*}} @"dfs$f1"
@f2 = alias void ()* @f1

; CHECK: @"dfs$g2" = alias {{.*}} @"dfs$g1"
@g2 = alias bitcast (void (i8*)* @g1 to void (i16*)*)

; CHECK: define void @"dfs$f1"
define void @f1() {
  ret void
}

; CHECK: define void @"dfs$g1"
define void @g1(i8*) {
  ret void
}

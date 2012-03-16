; RUN: opt < %s -S -instcombine | FileCheck %s                                                                                                                                                                          
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

%0 = type { i32, i8, i8, i8 }

@G = external hidden global %0, align 4

define void @f1(i64 %a1) nounwind ssp align 2 {
; CHECK: store i64 %a1, i64* bitcast (%0* @G to i64*), align 4                                                                                                                                                          
  store i64 %a1, i64* bitcast (%0* @G to i64*)
  ret void
}

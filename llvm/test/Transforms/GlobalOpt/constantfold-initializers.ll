; RUN: opt < %s -S -globalopt | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

@.str91250 = global [3 x i8] zeroinitializer

; CHECK: @A = global i1 false
@A = global i1 icmp ne (i64 sub nsw (i64 ptrtoint (i8* getelementptr inbounds ([3 x i8]* @.str91250, i64 0, i64 1) to i64), i64 ptrtoint ([3 x i8]* @.str91250 to i64)), i64 1)

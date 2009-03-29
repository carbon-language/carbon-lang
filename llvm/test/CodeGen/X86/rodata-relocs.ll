; RUN: llvm-as < %s | llc -relocation-model=static | grep rodata | count 3
; RUN: llvm-as < %s | llc -relocation-model=static | grep -F "rodata.cst" | count 2
; RUN: llvm-as < %s | llc -relocation-model=pic | grep rodata | count 2
; RUN: llvm-as < %s | llc -relocation-model=pic | grep -F ".data" | count 1

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
@a = internal constant [2 x i32] [i32 1, i32 2]		; <[2 x i32]*> [#uses=1]
@a1 = constant [2 x i32] [i32 1, i32 2]		; <[2 x i32]*> [#uses=1]
@e = internal constant [2 x [2 x i32]] [[2 x i32] [i32 1, i32 2], [2 x i32] [i32 3, i32 4]], align 16		; <[2 x [2 x i32]]*> [#uses=1]
@e1 = constant [2 x [2 x i32]] [[2 x i32] [i32 1, i32 2], [2 x i32] [i32 3, i32 4]], align 16		; <[2 x [2 x i32]]*> [#uses=1]
@p = constant i8* bitcast ([2 x i32]* @a to i8*)		; <i8**> [#uses=0]
@t = constant i8* bitcast ([2 x [2 x i32]]* @e to i8*)		; <i8**> [#uses=0]
@p1 = constant i8* bitcast ([2 x i32]* @a1 to i8*)		; <i8**> [#uses=0]
@t1 = constant i8* bitcast ([2 x [2 x i32]]* @e1 to i8*)		; <i8**> [#uses=0]

; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@v1 = global i32 0
; CHECK: @v1 = global i32 0

@v2 = global [1 x i32] zeroinitializer
; CHECK: @v2 = global [1 x i32] zeroinitializer

@v3 = alias i16, i32* @v1
; CHECK: @v3 = alias i16, i32* @v1

@v4 = alias i32, [1 x i32]* @v2
; CHECK: @v4 = alias i32, [1 x i32]* @v2

@v5 = alias addrspace(2) i32, i32* @v1
; CHECK: @v5 = alias addrspace(2) i32, i32* @v1

@v6 = alias i16, i32* @v1
; CHECK: @v6 = alias i16, i32* @v1

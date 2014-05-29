; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@v1 = global i32 0
; CHECK: @v1 = global i32 0

@v2 = global [1 x i32] zeroinitializer
; CHECK: @v2 = global [1 x i32] zeroinitializer

@a1 = alias i16, i32* @v1
; CHECK: @a1 = alias i16, i32* @v1

@a2 = alias i32, [1 x i32]* @v2
; CHECK: @a2 = alias i32, [1 x i32]* @v2

@a3 = alias addrspace(2) i32, i32* @v1
; CHECK: @a3 = alias addrspace(2) i32, i32* @v1

@a4 = alias i16, i32* @v1
; CHECK: @a4 = alias i16, i32* @v1

@a5 = thread_local(localdynamic) alias i32* @v1
; CHECK: @a5 = thread_local(localdynamic) alias i32* @v1

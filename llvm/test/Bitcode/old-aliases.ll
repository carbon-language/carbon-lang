; RUN: llvm-dis < %s.bc | FileCheck %s

; old-aliases.bc consist of this file assembled with an old llvm-as (3.5 trunk)
; from when aliases contained a ConstantExpr.

@v1 = global i32 0
; CHECK: @v1 = global i32 0

@v2 = global [1 x i32] zeroinitializer
; CHECK: @v2 = global [1 x i32] zeroinitializer

@v3 = alias bitcast (i32* @v1 to i16*)
; CHECK: @v3 = alias bitcast (i32* @v1 to i16*)

@v4 = alias getelementptr ([1 x i32]* @v2, i32 0, i32 0)
; CHECK: @v4 = alias getelementptr inbounds ([1 x i32]* @v2, i32 0, i32 0)

@v5 = alias i32 addrspace(2)* addrspacecast (i32 addrspace(0)* @v1 to i32 addrspace(2)*)
; CHECK: @v5 = alias addrspacecast (i32* @v1 to i32 addrspace(2)*)

@v6 = alias i16* @v3
; CHECK: @v6 = alias i16* @v3

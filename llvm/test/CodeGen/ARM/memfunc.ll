; RUN: llc < %s -march=arm -o - | FileCheck %s
; RUN: llc < %s -mtriple=arm-none-eabi -o - | FileCheck --check-prefix=EABI %s

@from = common global [500 x i32] zeroinitializer, align 4
@to = common global [500 x i32] zeroinitializer, align 4

define void @f() {
entry:

        ; CHECK: memmove
        ; EABI: __aeabi_memmove
        call void @llvm.memmove.i32( i8* bitcast ([500 x i32]* @from to i8*), i8* bitcast ([500 x i32]* @to to i8*), i32 500, i32 0 )

        ; CHECK: memcpy
        ; EABI: __aeabi_memcpy
        call void @llvm.memcpy.i32( i8* bitcast ([500 x i32]* @from to i8*), i8* bitcast ([500 x i32]* @to to i8*), i32 500, i32 0 )

        ; EABI memset swaps arguments
        ; CHECK: mov r1, #0
        ; CHECK: memset
        ; EABI: mov r2, #0
        ; EABI: __aeabi_memset
        call void @llvm.memset.i32( i8* bitcast ([500 x i32]* @from to i8*), i8 0, i32 500, i32 0 )
        unreachable
}

declare void @llvm.memmove.i32(i8*, i8*, i32, i32)

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

declare void @llvm.memset.i32(i8*, i8, i32, i32)


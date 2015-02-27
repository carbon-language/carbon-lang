; RUN: opt -codegenprepare -S < %s | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare zeroext i1 @return_i1()

define i32 @test_sor_basic(i32* %base) {
; CHECK: getelementptr i32, i32* %base, i32 15
; CHECK: getelementptr i32, i32* %base-new, i32 15
entry:
       %ptr = getelementptr i32, i32* %base, i32 15
       %tok = call i32 (i1 ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_i1f(i1 ()* @return_i1, i32 0, i32 0, i32 0, i32* %base, i32* %ptr)
       %base-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 4, i32 4)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 4, i32 5)
       %ret = load i32* %ptr-new
       ret i32 %ret
}

define i32 @test_sor_two_derived(i32* %base) {
; CHECK: getelementptr i32, i32* %base, i32 15
; CHECK: getelementptr i32, i32* %base, i32 12
; CHECK: getelementptr i32, i32* %base-new, i32 15
; CHECK: getelementptr i32, i32* %base-new, i32 12
entry:
       %ptr = getelementptr i32, i32* %base, i32 15
       %ptr2 = getelementptr i32, i32* %base, i32 12
       %tok = call i32 (i1 ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_i1f(i1 ()* @return_i1, i32 0, i32 0, i32 0, i32* %base, i32* %ptr, i32* %ptr2)
       %base-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 4, i32 4)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 4, i32 5)
       %ptr2-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 4, i32 6)
       %ret = load i32* %ptr-new
       ret i32 %ret
}

define i32 @test_sor_ooo(i32* %base) {
; CHECK: getelementptr i32, i32* %base, i32 15
; CHECK: getelementptr i32, i32* %base-new, i32 15
entry:
       %ptr = getelementptr i32, i32* %base, i32 15
       %tok = call i32 (i1 ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_i1f(i1 ()* @return_i1, i32 0, i32 0, i32 0, i32* %base, i32* %ptr)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 4, i32 5)
       %base-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 4, i32 4)
       %ret = load i32* %ptr-new
       ret i32 %ret
}

define i32 @test_sor_gep_smallint([3 x i32]* %base) {
; CHECK: getelementptr [3 x i32], [3 x i32]* %base, i32 0, i32 2
; CHECK: getelementptr [3 x i32], [3 x i32]* %base-new, i32 0, i32 2
entry:
       %ptr = getelementptr [3 x i32], [3 x i32]* %base, i32 0, i32 2
       %tok = call i32 (i1 ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_i1f(i1 ()* @return_i1, i32 0, i32 0, i32 0, [3 x i32]* %base, i32* %ptr)
       %base-new = call [3 x i32]* @llvm.experimental.gc.relocate.p0a3i32(i32 %tok, i32 4, i32 4)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 4, i32 5)
       %ret = load i32* %ptr-new
       ret i32 %ret
}

define i32 @test_sor_gep_largeint([3 x i32]* %base) {
; CHECK: getelementptr [3 x i32], [3 x i32]* %base, i32 0, i32 21
; CHECK-NOT: getelementptr [3 x i32], [3 x i32]* %base-new, i32 0, i32 21
entry:
       %ptr = getelementptr [3 x i32], [3 x i32]* %base, i32 0, i32 21
       %tok = call i32 (i1 ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_i1f(i1 ()* @return_i1, i32 0, i32 0, i32 0, [3 x i32]* %base, i32* %ptr)
       %base-new = call [3 x i32]* @llvm.experimental.gc.relocate.p0a3i32(i32 %tok, i32 4, i32 4)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 4, i32 5)
       %ret = load i32* %ptr-new
       ret i32 %ret
}

define i32 @test_sor_noop(i32* %base) {
; CHECK: getelementptr i32, i32* %base, i32 15
; CHECK: call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 4, i32 5)
; CHECK: call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 4, i32 6)
entry:
       %ptr = getelementptr i32, i32* %base, i32 15
       %ptr2 = getelementptr i32, i32* %base, i32 12
       %tok = call i32 (i1 ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_i1f(i1 ()* @return_i1, i32 0, i32 0, i32 0, i32* %base, i32* %ptr, i32* %ptr2)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 4, i32 5)
       %ptr2-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 4, i32 6)
       %ret = load i32* %ptr-new
       ret i32 %ret
}

declare i32 @llvm.experimental.gc.statepoint.p0f_i1f(i1 ()*, i32, i32, ...)
declare i32* @llvm.experimental.gc.relocate.p0i32(i32, i32, i32)
declare [3 x i32]* @llvm.experimental.gc.relocate.p0a3i32(i32, i32, i32)

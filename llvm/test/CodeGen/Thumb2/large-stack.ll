; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define void @test1() {
; CHECK: test1:
; CHECK: sub sp, #64 * 4
    %tmp = alloca [ 64 x i32 ] , align 4
    ret void
}

define void @test2() {
; CHECK: test2:
; CHECK: sub.w sp, sp, #4160
; CHECK: sub sp, #2 * 4
    %tmp = alloca [ 4168 x i8 ] , align 4
    ret void
}

define i32 @test3() {
; CHECK: test3:
; CHECK: sub.w sp, sp, #805306368
; CHECK: sub sp, #6 * 4
    %retval = alloca i32, align 4
    %tmp = alloca i32, align 4
    %a = alloca [805306369 x i8], align 16
    store i32 0, i32* %tmp
    %tmp1 = load i32* %tmp
    ret i32 %tmp1
}

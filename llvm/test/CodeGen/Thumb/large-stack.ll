; RUN: llc < %s -mtriple=thumb-apple-ios | FileCheck %s

define void @test1() {
; CHECK-LABEL: test1:
; CHECK: sub sp, #256
; CHECK: add sp, #256
    %tmp = alloca [ 64 x i32 ] , align 4
    ret void
}

define void @test2() {
; CHECK-LABEL: test2:
; CHECK: ldr r0, LCPI
; CHECK: add sp, r0
; CHECK: subs r4, r7, #4
; CHECK: mov sp, r4
    %tmp = alloca [ 4168 x i8 ] , align 4
    ret void
}

define i32 @test3() {
; CHECK-LABEL: test3:
; CHECK: ldr r1, LCPI
; CHECK: add sp, r1
; CHECK: ldr r1, LCPI
; CHECK: add r1, sp
; CHECK: subs r4, r7, #4
; CHECK: mov sp, r4
    %retval = alloca i32, align 4
    %tmp = alloca i32, align 4
    %a = alloca [805306369 x i8], align 16
    store i32 0, i32* %tmp
    %tmp1 = load i32* %tmp
    ret i32 %tmp1
}

; Here, the adds get optimized out because they are dead, but the calculation
; of the address of stack_a is dead but not optimized out. When the address
; calculation gets expanded to two instructions, we need to avoid reading a
; dead register.
; No CHECK lines (just test for crashes), as we hope this will be optimised
; better in future.
define i32 @test4() {
entry:
  %stack_a = alloca i8, align 1
  %stack_b = alloca [256 x i32*], align 4
  %int = ptrtoint i8* %stack_a to i32
  %add = add i32 %int, 1
  br label %block2

block2:
  %add2 = add i32 %add, 1
  ret i32 0
}

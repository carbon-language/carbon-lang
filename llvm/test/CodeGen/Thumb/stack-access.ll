; RUN: llc -mtriple=thumb-eabi < %s -o - | FileCheck %s

; Check that stack addresses are generated using a single ADD
define void @test1(i8** %p) {
  %x = alloca i8, align 1
  %y = alloca i8, align 1
  %z = alloca i8, align 1
; CHECK: add r1, sp, #8
; CHECK: str r1, [r0]
  store i8* %x, i8** %p, align 4
; CHECK: add r1, sp, #4
; CHECK: str r1, [r0]
  store i8* %y, i8** %p, align 4
; CHECK: mov r1, sp
; CHECK: str r1, [r0]
  store i8* %z, i8** %p, align 4
  ret void
}

; Stack offsets larger than 1020 still need two ADDs
define void @test2([1024 x i8]** %p) {
  %arr1 = alloca [1024 x i8], align 1
  %arr2 = alloca [1024 x i8], align 1
; CHECK: add r1, sp, #1020
; CHECK: adds r1, #4
; CHECK: str r1, [r0]
  store [1024 x i8]* %arr1, [1024 x i8]** %p, align 4
; CHECK: mov r1, sp
; CHECK: str r1, [r0]
  store [1024 x i8]* %arr2, [1024 x i8]** %p, align 4
  ret void
}

; If possible stack-based lrdb/ldrh are widened to use SP-based addressing
define i32 @test3() #0 {
  %x = alloca i8, align 1
  %y = alloca i8, align 1
; CHECK: ldr r0, [sp]
  %1 = load i8, i8* %x, align 1
; CHECK: ldr r1, [sp, #4]
  %2 = load i8, i8* %y, align 1
  %3 = add nsw i8 %1, %2
  %4 = zext i8 %3 to i32
  ret i32 %4
}

define i32 @test4() #0 {
  %x = alloca i16, align 2
  %y = alloca i16, align 2
; CHECK: ldr r0, [sp]
  %1 = load i16, i16* %x, align 2
; CHECK: ldr r1, [sp, #4]
  %2 = load i16, i16* %y, align 2
  %3 = add nsw i16 %1, %2
  %4 = zext i16 %3 to i32
  ret i32 %4
}

; Don't widen if the value needs to be zero-extended
define zeroext i8 @test5() {
  %x = alloca i8, align 1
; CHECK: mov r0, sp
; CHECK: ldrb r0, [r0]
  %1 = load i8, i8* %x, align 1
  ret i8 %1
}

define zeroext i16 @test6() {
  %x = alloca i16, align 2
; CHECK: mov r0, sp
; CHECK: ldrh r0, [r0]
  %1 = load i16, i16* %x, align 2
  ret i16 %1
}

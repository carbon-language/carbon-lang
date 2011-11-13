; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-darwin | FileCheck %s --check-prefix=THUMB

@message1 = global [60 x i8] c"The LLVM Compiler Infrastructure\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 1
@temp = common global [60 x i8] zeroinitializer, align 1

define void @t1() nounwind ssp {
; ARM: t1
; ARM: ldr r0, LCPI0_0
; ARM: add r0, r0, #5
; ARM: movw r1, #64
; ARM: movw r2, #10
; ARM: uxtb r1, r1
; ARM: bl #14
; THUMB: t1
; THUMB: ldr.n r0, LCPI0_0
; THUMB: adds r0, #5
; THUMB: movs r1, #64
; THUMB: movt r1, #0
; THUMB: movs r2, #10
; THUMB: movt r2, #0
; THUMB: uxtb r1, r1
; THUMB: bl _memset
  call void @llvm.memset.p0i8.i32(i8* getelementptr inbounds ([60 x i8]* @message1, i32 0, i32 5), i8 64, i32 10, i32 1, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind

define void @t2() nounwind ssp {
; ARM: t2
; ARM: ldr r0, LCPI1_0
; ARM: ldr r0, [r0]
; ARM: add r1, r0, #4
; ARM: add r0, r0, #16
; ARM: movw r2, #10
; ARM: str r0, [sp]                @ 4-byte Spill
; ARM: mov r0, r1
; ARM: ldr r1, [sp]                @ 4-byte Reload
; ARM: bl #14
; THUMB: t2
; THUMB: ldr.n r0, LCPI1_0
; THUMB: ldr r0, [r0]
; THUMB: adds r1, r0, #4
; THUMB: adds r0, #16
; THUMB: movs r2, #10
; THUMB: movt r2, #0
; THUMB: mov r0, r1
; THUMB: bl _memcpy
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* getelementptr inbounds ([60 x i8]* @temp, i32 0, i32 4), i8* getelementptr inbounds ([60 x i8]* @temp, i32 0, i32 16), i32 10, i32 1, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

define void @t3() nounwind ssp {
; ARM: t3
; ARM: ldr r0, LCPI2_0
; ARM: ldr r0, [r0]
; ARM: add r1, r0, #4
; ARM: add r0, r0, #16
; ARM: movw r2, #10
; ARM: mov r0, r1
; ARM: bl #14
; THUMB: t3
; THUMB: ldr.n r0, LCPI2_0
; THUMB: ldr r0, [r0]
; THUMB: adds r1, r0, #4
; THUMB: adds r0, #16
; THUMB: movs r2, #10
; THUMB: movt r2, #0
; THUMB: mov r0, r1
; THUMB: bl _memmove
  call void @llvm.memmove.p0i8.p0i8.i32(i8* getelementptr inbounds ([60 x i8]* @temp, i32 0, i32 4), i8* getelementptr inbounds ([60 x i8]* @temp, i32 0, i32 16), i32 10, i32 1, i1 false)
  ret void
}

declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

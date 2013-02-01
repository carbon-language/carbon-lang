; RUN: llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s
declare void @use_addr(i8*)

@addr = global i8* null

define void @test_bigframe() {
; CHECK: test_bigframe:

  %var1 = alloca i8, i32 20000000
  %var2 = alloca i8, i32 16
  %var3 = alloca i8, i32 20000000
; CHECK: sub sp, sp, #496
; CHECK: str x30, [sp, #488]
; CHECK: ldr [[FRAMEOFFSET:x[0-9]+]], [[FRAMEOFFSET_CPI:.LCPI0_[0-9]+]]
; CHECK: sub sp, sp, [[FRAMEOFFSET]]

; CHECK: ldr [[VAR1OFFSET:x[0-9]+]], [[VAR1LOC_CPI:.LCPI0_[0-9]+]]
; CHECK: add {{x[0-9]+}}, sp, [[VAR1OFFSET]]
  store volatile i8* %var1, i8** @addr

  %var1plus2 = getelementptr i8* %var1, i32 2
  store volatile i8* %var1plus2, i8** @addr

; CHECK: ldr [[VAR2OFFSET:x[0-9]+]], [[VAR2LOC_CPI:.LCPI0_[0-9]+]]
; CHECK: add {{x[0-9]+}}, sp, [[VAR2OFFSET]]
  store volatile i8* %var2, i8** @addr

  %var2plus2 = getelementptr i8* %var2, i32 2
  store volatile i8* %var2plus2, i8** @addr

  store volatile i8* %var3, i8** @addr

  %var3plus2 = getelementptr i8* %var3, i32 2
  store volatile i8* %var3plus2, i8** @addr

; CHECK: ldr [[FRAMEOFFSET:x[0-9]+]], [[FRAMEOFFSET_CPI]]
; CHECK: add sp, sp, [[FRAMEOFFSET]]
  ret void

; CHECK: [[FRAMEOFFSET_CPI]]:
; CHECK-NEXT: 39999536

; CHECK: [[VAR1LOC_CPI]]:
; CHECK-NEXT: 20000024

; CHECK: [[VAR2LOC_CPI]]:
; CHECK-NEXT: 20000008
}

define void @test_mediumframe() {
; CHECK: test_mediumframe:
  %var1 = alloca i8, i32 1000000
  %var2 = alloca i8, i32 16
  %var3 = alloca i8, i32 1000000
; CHECK: sub sp, sp, #496
; CHECK: str x30, [sp, #488]
; CHECK: sub sp, sp, #688
; CHECK-NEXT: sub sp, sp, #488, lsl #12

  store volatile i8* %var1, i8** @addr
; CHECK: add [[VAR1ADDR:x[0-9]+]], sp, #600
; CHECK: add [[VAR1ADDR]], [[VAR1ADDR]], #244, lsl #12

  %var1plus2 = getelementptr i8* %var1, i32 2
  store volatile i8* %var1plus2, i8** @addr
; CHECK: add [[VAR1PLUS2:x[0-9]+]], {{x[0-9]+}}, #2

  store volatile i8* %var2, i8** @addr
; CHECK: add [[VAR2ADDR:x[0-9]+]], sp, #584
; CHECK: add [[VAR2ADDR]], [[VAR2ADDR]], #244, lsl #12

  %var2plus2 = getelementptr i8* %var2, i32 2
  store volatile i8* %var2plus2, i8** @addr
; CHECK: add [[VAR2PLUS2:x[0-9]+]], {{x[0-9]+}}, #2

  store volatile i8* %var3, i8** @addr

  %var3plus2 = getelementptr i8* %var3, i32 2
  store volatile i8* %var3plus2, i8** @addr

; CHECK: add sp, sp, #688
; CHECK: add sp, sp, #488, lsl #12
; CHECK: ldr x30, [sp, #488]
; CHECK: add sp, sp, #496
  ret void
}


@bigspace = global [8 x i64] zeroinitializer

; If temporary registers are allocated for adjustment, they should *not* clobber
; argument registers.
define void @test_tempallocation([8 x i64] %val) nounwind {
; CHECK: test_tempallocation:
  %var = alloca i8, i32 1000000
; CHECK: sub sp, sp,

; Make sure the prologue is reasonably efficient
; CHECK-NEXT: stp x29, x30, [sp,
; CHECK-NEXT: stp x25, x26, [sp,
; CHECK-NEXT: stp x23, x24, [sp,
; CHECK-NEXT: stp x21, x22, [sp,
; CHECK-NEXT: stp x19, x20, [sp,

; Make sure we don't trash an argument register
; CHECK-NOT: ldr {{x[0-7]}}, .LCPI1
; CHECK: sub sp, sp,

; CHECK-NOT: ldr {{x[0-7]}}, .LCPI1

; CHECK: bl use_addr
  call void @use_addr(i8* %var)

  store [8 x i64] %val, [8 x i64]* @bigspace
  ret void
; CHECK: ret
}

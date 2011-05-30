; ModuleID = 'test1.c'
; RUN: opt -S -instcombine < %s | FileCheck %s
target triple = "x86_64-apple-macosx10.6.6"

define zeroext i16 @foo(i32 %on_off, i16* %puls) nounwind uwtable ssp {
entry:
  %on_off.addr = alloca i32, align 4
  %puls.addr = alloca i16*, align 8
  %a = alloca i32, align 4
  store i32 %on_off, i32* %on_off.addr, align 4
  store i16* %puls, i16** %puls.addr, align 8
  %tmp = load i32* %on_off.addr, align 4
; CHECK-NOT: sub
; CHECK-NOT: mul
; (1 - %tmp) * (-2) -> (%tmp - 1) * 2
  %sub = sub i32 1, %tmp
  %mul = mul i32 %sub, -2
; CHECK: shl
; CHECK-NEXT: add
  store i32 %mul, i32* %a, align 4
  %tmp1 = load i32* %a, align 4
  %conv = trunc i32 %tmp1 to i16
  ret i16 %conv
}

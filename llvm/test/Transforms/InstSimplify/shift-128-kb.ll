; RUN: opt -S -instsimplify < %s | FileCheck %s

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define zeroext i1 @_Z10isNegativemj(i64 %Val, i32 zeroext %IntegerBitWidth) {
entry:
  %conv = zext i32 %IntegerBitWidth to i64
  %sub = sub i64 128, %conv
  %conv1 = trunc i64 %sub to i32
  %conv2 = zext i64 %Val to i128
  %sh_prom = zext i32 %conv1 to i128
  %shl = shl i128 %conv2, %sh_prom
  %shr = ashr i128 %shl, %sh_prom
  %cmp = icmp slt i128 %shr, 0
  ret i1 %cmp
}

; CHECK-LABEL: @_Z10isNegativemj
; CHECK-NOT: ret i1 false
; CHECK: ret i1 %cmp


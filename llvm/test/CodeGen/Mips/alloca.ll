; RUN: llc -march=mipsel -mcpu=4ke < %s | FileCheck %s

define i32 @twoalloca(i32 %size) nounwind {
entry:
; CHECK: subu  $[[T0:[0-9]+]], $sp, $[[SZ:[0-9]+]]
; CHECK: addu  $sp, $zero, $[[T0]]
; CHECK: addu  $[[SP1:[0-9]+]], $zero, $sp
; CHECK: subu  $[[T1:[0-9]+]], $sp, $[[SZ]]
; CHECK: addu  $sp, $zero, $[[T1]]
; CHECK: addu  $[[SP2:[0-9]+]], $zero, $sp
; CHECK: lw  $25, %call16(foo)($gp)
; CHECK: addiu $4, $[[SP1]], 24
; CHECK: jalr  $25
; CHECK: lw  $25, %call16(foo)($gp)
; CHECK: addiu $4, $[[SP2]], 24
; CHECK: jalr  $25
  %tmp1 = alloca i8, i32 %size, align 4
  %add.ptr = getelementptr inbounds i8* %tmp1, i32 5
  store i8 97, i8* %add.ptr, align 1
  %tmp4 = alloca i8, i32 %size, align 4
  call void @foo2(double 1.000000e+00, double 2.000000e+00, i32 3) nounwind
  %call = call i32 @foo(i8* %tmp1) nounwind
  %call7 = call i32 @foo(i8* %tmp4) nounwind
  %add = add nsw i32 %call7, %call
  ret i32 %add
}

declare void @foo2(double, double, i32)

declare i32 @foo(i8*)


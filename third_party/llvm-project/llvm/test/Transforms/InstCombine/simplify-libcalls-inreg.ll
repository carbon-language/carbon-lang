; RUN: opt -passes=instcombine -S %s | FileCheck %s

; The intent of this test is to check that the declarations produces for
; libcalls retains the inreg parameter attribute.

target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

declare ptr @foo()
declare i32 @memcmp(ptr inreg nocapture noundef, ptr inreg nocapture noundef, i32 inreg noundef)
declare i32 @printf(i8*, ...)
declare double @exp2(double)
declare i32 @__sprintf_chk(i8*, i32, i32, i8*, ...)
@a = common global [60 x i8] zeroinitializer, align 1
@b = common global [60 x i8] zeroinitializer, align 1
@h = constant [2 x i8] c"h\00"

; CHECK:     declare i32 @bcmp(ptr inreg nocapture, ptr inreg nocapture, i32 inreg)
; CHECK-NOT: declare i32 @bcmp(ptr nocapture, ptr nocapture, i32)

define i32 @baz(ptr inreg noundef %s2, i32 inreg noundef %n){
  %call = call ptr @foo()
  %call1 = call i32 @memcmp(ptr inreg noundef %call, ptr inreg noundef %s2, i32 inreg noundef %n)
  %cmp = icmp eq i32 %call1, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK:     declare noundef i32 @putchar(i32 inreg noundef)
; CHECK-NOT: declare noundef i32 @putchar(i32 noundef)

define void @test_fewer_params_than_num_register_parameters() {
  %fmt = getelementptr [2 x i8], [2 x i8]* @h, i32 0, i32 0
  call i32 (i8*, ...) @printf(i8* %fmt)
  ret void
}

; CHECK:     declare double @ldexp(double, i32 inreg)
; CHECK-NOT: declare double @ldexp(double, i32)

define double @test_non_int_params(i16 signext %x) {
  %conv = sitofp i16 %x to double
  %ret = call double @exp2(double %conv)
  ret double %ret
}

; CHECK:     declare noundef i32 @sprintf(ptr noalias nocapture noundef writeonly, ptr nocapture noundef readonly, ...)
; CHECK-NOT: declare noundef i32 @sprintf(ptr inreg noalias nocapture noundef writeonly, ptr inreg nocapture noundef readonly, ...)
define i32 @test_variadic() {
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %fmt = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i32 (i8*, i32, i32, i8*, ...) @__sprintf_chk(i8* %dst, i32 0, i32 -1, i8* %fmt)
  ret i32 %ret
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"NumRegisterParameters", i32 3}

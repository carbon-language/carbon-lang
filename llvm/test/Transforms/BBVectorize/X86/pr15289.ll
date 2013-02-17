; RUN: opt < %s -basicaa -bb-vectorize -disable-output
; This is a bugpoint-reduced test case. It did not always assert, but does reproduce the bug
; and running under valgrind (or some similar tool) will catch the error.

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin12.2.0"

%0 = type { [10 x { float, float }], [10 x { float, float }], [10 x { float, float }], [10 x { float, float }], [10 x { float, float }] }
%1 = type { [10 x [8 x i8]] }
%2 = type { i64, i64 }
%3 = type { [10 x i64], i64, i64, i64, i64, i64 }
%4 = type { i64, i64, i64, i64, i64, i64 }
%5 = type { [10 x i64] }
%6 = type { [10 x float], [10 x float], [10 x float], [10 x float] }
%struct.__st_parameter_dt.1.3.5.7 = type { %struct.__st_parameter_common.0.2.4.6, i64, i64*, i64*, i8*, i8*, i32, i32, i8*, i8*, i32, i32, i8*, [256 x i8], i32*, i64, i8*, i32, i32, i8*, i8*, i32, i32, i8*, i8*, i32, i32, i8*, i8*, i32, [4 x i8] }
%struct.__st_parameter_common.0.2.4.6 = type { i32, i32, i8*, i32, i32, i8*, i32* }

@cctenso_ = external unnamed_addr global %0, align 32
@ctenso_ = external unnamed_addr global %1, align 32
@i_dim_ = external unnamed_addr global %2, align 16
@itenso1_ = external unnamed_addr global %3, align 32
@itenso2_ = external unnamed_addr global %4, align 32
@ltenso_ = external unnamed_addr global %5, align 32
@rtenso_ = external unnamed_addr global %6, align 32
@.cst = external unnamed_addr constant [8 x i8], align 8
@.cst1 = external unnamed_addr constant [3 x i8], align 8
@.cst2 = external unnamed_addr constant [29 x i8], align 8
@.cst3 = external unnamed_addr constant [32 x i8], align 64

define void @cart_to_dc2y_(double* noalias nocapture %xx, double* noalias nocapture %yy, double* noalias nocapture %zz, [5 x { double, double }]* noalias nocapture %c2ten) nounwind uwtable {
entry:
  %0 = fmul double undef, undef
  %1 = fmul double undef, undef
  %2 = fadd double undef, undef
  %3 = fmul double undef, 0x3FE8B8B76E3E9919
  %4 = fsub double %0, %1
  %5 = fsub double -0.000000e+00, undef
  %6 = fmul double undef, undef
  %7 = fmul double %4, %6
  %8 = fmul double undef, 2.000000e+00
  %9 = fmul double %8, undef
  %10 = fmul double undef, %9
  %11 = fmul double %10, undef
  %12 = fsub double undef, %7
  %13 = fmul double %3, %12
  %14 = fmul double %3, undef
  %15 = getelementptr inbounds [5 x { double, double }]* %c2ten, i64 0, i64 0, i32 0
  store double %13, double* %15, align 8, !tbaa !0
  %16 = getelementptr inbounds [5 x { double, double }]* %c2ten, i64 0, i64 0, i32 1
  %17 = fmul double undef, %8
  %18 = fmul double %17, undef
  %19 = fmul double undef, %18
  %20 = fadd double undef, undef
  %21 = fmul double %3, %19
  %22 = fsub double -0.000000e+00, %21
  %23 = getelementptr inbounds [5 x { double, double }]* %c2ten, i64 0, i64 1, i32 0
  store double %22, double* %23, align 8, !tbaa !0
  %24 = getelementptr inbounds [5 x { double, double }]* %c2ten, i64 0, i64 1, i32 1
  %25 = fmul double undef, 0x3FE42F601A8C6794
  %26 = fmul double undef, 2.000000e+00
  %27 = fsub double %26, %0
  %28 = fmul double %6, undef
  %29 = fsub double undef, %28
  %30 = getelementptr inbounds [5 x { double, double }]* %c2ten, i64 0, i64 2, i32 0
  store double undef, double* %30, align 8, !tbaa !0
  %31 = getelementptr inbounds [5 x { double, double }]* %c2ten, i64 0, i64 2, i32 1
  %32 = fmul double undef, %17
  %33 = fmul double undef, %17
  %34 = fmul double undef, %32
  %35 = fmul double undef, %33
  %36 = fsub double undef, %35
  %37 = fmul double %3, %34
  %38 = getelementptr inbounds [5 x { double, double }]* %c2ten, i64 0, i64 3, i32 0
  store double %37, double* %38, align 8, !tbaa !0
  %39 = getelementptr inbounds [5 x { double, double }]* %c2ten, i64 0, i64 3, i32 1
  %40 = fmul double undef, %8
  %41 = fmul double undef, %40
  %42 = fmul double undef, %41
  %43 = fsub double undef, %42
  %44 = fmul double %3, %43
  %45 = getelementptr inbounds [5 x { double, double }]* %c2ten, i64 0, i64 4, i32 0
  store double %13, double* %45, align 8, !tbaa !0
  %46 = getelementptr inbounds [5 x { double, double }]* %c2ten, i64 0, i64 4, i32 1
  %47 = fsub double -0.000000e+00, %14
  store double %47, double* %16, align 8, !tbaa !0
  store double undef, double* %24, align 8, !tbaa !0
  store double -0.000000e+00, double* %31, align 8, !tbaa !0
  store double undef, double* %39, align 8, !tbaa !0
  store double undef, double* %46, align 8, !tbaa !0
  ret void
}

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!0 = metadata !{metadata !"alias set 17: real(kind=8)", metadata !1}
!1 = metadata !{metadata !1}

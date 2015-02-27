; RUN: opt < %s -O1 -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;CHECK: load <2 x double>, <2 x double>*
;CHECK: fadd <2 x double>
;CHECK: store <2 x double>

; Function Attrs: nounwind uwtable
define void @foo(double* %x) #0 {
  %1 = alloca double*, align 8
  store double* %x, double** %1, align 8
  %2 = load double*, double** %1, align 8
  %3 = getelementptr inbounds double, double* %2, i64 0
  %4 = load double, double* %3, align 8
  %5 = load double*, double** %1, align 8
  %6 = getelementptr inbounds double, double* %5, i64 0
  %7 = load double, double* %6, align 8
  %8 = fadd double %4, %7
  %9 = load double*, double** %1, align 8
  %10 = getelementptr inbounds double, double* %9, i64 0
  %11 = load double, double* %10, align 8
  %12 = fadd double %8, %11
  %13 = load double*, double** %1, align 8
  %14 = getelementptr inbounds double, double* %13, i64 0
  store double %12, double* %14, align 8
  %15 = load double*, double** %1, align 8
  %16 = getelementptr inbounds double, double* %15, i64 1
  %17 = load double, double* %16, align 8
  %18 = load double*, double** %1, align 8
  %19 = getelementptr inbounds double, double* %18, i64 1
  %20 = load double, double* %19, align 8
  %21 = fadd double %17, %20
  %22 = load double*, double** %1, align 8
  %23 = getelementptr inbounds double, double* %22, i64 1
  %24 = load double, double* %23, align 8
  %25 = fadd double %21, %24
  %26 = load double*, double** %1, align 8
  %27 = getelementptr inbounds double, double* %26, i64 1
  store double %25, double* %27, align 8
  %28 = load double*, double** %1, align 8
  %29 = getelementptr inbounds double, double* %28, i64 2
  %30 = load double, double* %29, align 8
  %31 = load double*, double** %1, align 8
  %32 = getelementptr inbounds double, double* %31, i64 2
  %33 = load double, double* %32, align 8
  %34 = fadd double %30, %33
  %35 = load double*, double** %1, align 8
  %36 = getelementptr inbounds double, double* %35, i64 2
  %37 = load double, double* %36, align 8
  %38 = fadd double %34, %37
  %39 = load double*, double** %1, align 8
  %40 = getelementptr inbounds double, double* %39, i64 2
  store double %38, double* %40, align 8
  %41 = load double*, double** %1, align 8
  %42 = getelementptr inbounds double, double* %41, i64 3
  %43 = load double, double* %42, align 8
  %44 = load double*, double** %1, align 8
  %45 = getelementptr inbounds double, double* %44, i64 3
  %46 = load double, double* %45, align 8
  %47 = fadd double %43, %46
  %48 = load double*, double** %1, align 8
  %49 = getelementptr inbounds double, double* %48, i64 3
  %50 = load double, double* %49, align 8
  %51 = fadd double %47, %50
  %52 = load double*, double** %1, align 8
  %53 = getelementptr inbounds double, double* %52, i64 3
  store double %51, double* %53, align 8
  ret void
}

attributes #0 = { nounwind }

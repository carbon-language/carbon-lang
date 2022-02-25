; RUN: llc < %s -march=sparc -no-integrated-as
; ModuleID = 'mult-alt-generic.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32"
target triple = "sparc"

@mout0 = common global i32 0, align 4
@min1 = common global i32 0, align 4
@marray = common global [2 x i32] zeroinitializer, align 4

define void @single_m() nounwind {
entry:
  call void asm "foo $1,$0", "=*m,*m"(i32* @mout0, i32* @min1) nounwind
  ret void
}

define void @single_o() nounwind {
entry:
  %out0 = alloca i32, align 4
  %index = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  store i32 1, i32* %index, align 4
  ret void
}

define void @single_V() nounwind {
entry:
  ret void
}

define void @single_lt() nounwind {
entry:
  %out0 = alloca i32, align 4
  %in1 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  store i32 1, i32* %in1, align 4
  %tmp = load i32, i32* %in1, align 4
  %0 = call i32 asm "foo $1,$0", "=r,<r"(i32 %tmp) nounwind
  store i32 %0, i32* %out0, align 4
  %tmp1 = load i32, i32* %in1, align 4
  %1 = call i32 asm "foo $1,$0", "=r,r<"(i32 %tmp1) nounwind
  store i32 %1, i32* %out0, align 4
  ret void
}

define void @single_gt() nounwind {
entry:
  %out0 = alloca i32, align 4
  %in1 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  store i32 1, i32* %in1, align 4
  %tmp = load i32, i32* %in1, align 4
  %0 = call i32 asm "foo $1,$0", "=r,>r"(i32 %tmp) nounwind
  store i32 %0, i32* %out0, align 4
  %tmp1 = load i32, i32* %in1, align 4
  %1 = call i32 asm "foo $1,$0", "=r,r>"(i32 %tmp1) nounwind
  store i32 %1, i32* %out0, align 4
  ret void
}

define void @single_r() nounwind {
entry:
  %out0 = alloca i32, align 4
  %in1 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  store i32 1, i32* %in1, align 4
  %tmp = load i32, i32* %in1, align 4
  %0 = call i32 asm "foo $1,$0", "=r,r"(i32 %tmp) nounwind
  store i32 %0, i32* %out0, align 4
  ret void
}

define void @single_i() nounwind {
entry:
  %out0 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  %0 = call i32 asm "foo $1,$0", "=r,i"(i32 1) nounwind
  store i32 %0, i32* %out0, align 4
  ret void
}

define void @single_n() nounwind {
entry:
  %out0 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  %0 = call i32 asm "foo $1,$0", "=r,n"(i32 1) nounwind
  store i32 %0, i32* %out0, align 4
  ret void
}

define void @single_E() nounwind {
entry:
  %out0 = alloca double, align 8
  store double 0.000000e+000, double* %out0, align 8
; No lowering support.
;  %0 = call double asm "foo $1,$0", "=r,E"(double 1.000000e+001) nounwind
;  store double %0, double* %out0, align 8
  ret void
}

define void @single_F() nounwind {
entry:
  %out0 = alloca double, align 8
  store double 0.000000e+000, double* %out0, align 8
; No lowering support.
;  %0 = call double asm "foo $1,$0", "=r,F"(double 1.000000e+000) nounwind
;  store double %0, double* %out0, align 8
  ret void
}

define void @single_s() nounwind {
entry:
  %out0 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  ret void
}

define void @single_g() nounwind {
entry:
  %out0 = alloca i32, align 4
  %in1 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  store i32 1, i32* %in1, align 4
  %tmp = load i32, i32* %in1, align 4
  %0 = call i32 asm "foo $1,$0", "=r,imr"(i32 %tmp) nounwind
  store i32 %0, i32* %out0, align 4
  %tmp1 = load i32, i32* @min1, align 4
  %1 = call i32 asm "foo $1,$0", "=r,imr"(i32 %tmp1) nounwind
  store i32 %1, i32* %out0, align 4
  %2 = call i32 asm "foo $1,$0", "=r,imr"(i32 1) nounwind
  store i32 %2, i32* %out0, align 4
  ret void
}

define void @single_X() nounwind {
entry:
  %out0 = alloca i32, align 4
  %in1 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  store i32 1, i32* %in1, align 4
  %tmp = load i32, i32* %in1, align 4
  %0 = call i32 asm "foo $1,$0", "=r,X"(i32 %tmp) nounwind
  store i32 %0, i32* %out0, align 4
  %tmp1 = load i32, i32* @min1, align 4
  %1 = call i32 asm "foo $1,$0", "=r,X"(i32 %tmp1) nounwind
  store i32 %1, i32* %out0, align 4
  %2 = call i32 asm "foo $1,$0", "=r,X"(i32 1) nounwind
  store i32 %2, i32* %out0, align 4
  %3 = call i32 asm "foo $1,$0", "=r,X"(i32* getelementptr inbounds ([2 x i32], [2 x i32]* @marray, i32 0, i32 0)) nounwind
  store i32 %3, i32* %out0, align 4
; No lowering support.
;  %4 = call i32 asm "foo $1,$0", "=r,X"(double 1.000000e+001) nounwind
;  store i32 %4, i32* %out0, align 4
;  %5 = call i32 asm "foo $1,$0", "=r,X"(double 1.000000e+000) nounwind
;  store i32 %5, i32* %out0, align 4
  ret void
}

define void @single_p() nounwind {
entry:
  %out0 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  %0 = call i32 asm "foo $1,$0", "=r,r"(i32* getelementptr inbounds ([2 x i32], [2 x i32]* @marray, i32 0, i32 0)) nounwind
  store i32 %0, i32* %out0, align 4
  ret void
}

define void @multi_m() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  call void asm "foo $1,$0", "=*m|r,m|r"(i32* @mout0, i32 %tmp) nounwind
  ret void
}

define void @multi_o() nounwind {
entry:
  %out0 = alloca i32, align 4
  %index = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  store i32 1, i32* %index, align 4
  ret void
}

define void @multi_V() nounwind {
entry:
  ret void
}

define void @multi_lt() nounwind {
entry:
  %out0 = alloca i32, align 4
  %in1 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  store i32 1, i32* %in1, align 4
  %tmp = load i32, i32* %in1, align 4
  %0 = call i32 asm "foo $1,$0", "=r|r,r|<r"(i32 %tmp) nounwind
  store i32 %0, i32* %out0, align 4
  %tmp1 = load i32, i32* %in1, align 4
  %1 = call i32 asm "foo $1,$0", "=r|r,r|r<"(i32 %tmp1) nounwind
  store i32 %1, i32* %out0, align 4
  ret void
}

define void @multi_gt() nounwind {
entry:
  %out0 = alloca i32, align 4
  %in1 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  store i32 1, i32* %in1, align 4
  %tmp = load i32, i32* %in1, align 4
  %0 = call i32 asm "foo $1,$0", "=r|r,r|>r"(i32 %tmp) nounwind
  store i32 %0, i32* %out0, align 4
  %tmp1 = load i32, i32* %in1, align 4
  %1 = call i32 asm "foo $1,$0", "=r|r,r|r>"(i32 %tmp1) nounwind
  store i32 %1, i32* %out0, align 4
  ret void
}

define void @multi_r() nounwind {
entry:
  %out0 = alloca i32, align 4
  %in1 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  store i32 1, i32* %in1, align 4
  %tmp = load i32, i32* %in1, align 4
  %0 = call i32 asm "foo $1,$0", "=r|r,r|m"(i32 %tmp) nounwind
  store i32 %0, i32* %out0, align 4
  ret void
}

define void @multi_i() nounwind {
entry:
  %out0 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  %0 = call i32 asm "foo $1,$0", "=r|r,r|i"(i32 1) nounwind
  store i32 %0, i32* %out0, align 4
  ret void
}

define void @multi_n() nounwind {
entry:
  %out0 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  %0 = call i32 asm "foo $1,$0", "=r|r,r|n"(i32 1) nounwind
  store i32 %0, i32* %out0, align 4
  ret void
}

define void @multi_E() nounwind {
entry:
  %out0 = alloca double, align 8
  store double 0.000000e+000, double* %out0, align 8
; No lowering support.
;  %0 = call double asm "foo $1,$0", "=r|r,r|E"(double 1.000000e+001) nounwind
;  store double %0, double* %out0, align 8
  ret void
}

define void @multi_F() nounwind {
entry:
  %out0 = alloca double, align 8
  store double 0.000000e+000, double* %out0, align 8
; No lowering support.
;  %0 = call double asm "foo $1,$0", "=r|r,r|F"(double 1.000000e+000) nounwind
;  store double %0, double* %out0, align 8
  ret void
}

define void @multi_s() nounwind {
entry:
  %out0 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  ret void
}

define void @multi_g() nounwind {
entry:
  %out0 = alloca i32, align 4
  %in1 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  store i32 1, i32* %in1, align 4
  %tmp = load i32, i32* %in1, align 4
  %0 = call i32 asm "foo $1,$0", "=r|r,r|imr"(i32 %tmp) nounwind
  store i32 %0, i32* %out0, align 4
  %tmp1 = load i32, i32* @min1, align 4
  %1 = call i32 asm "foo $1,$0", "=r|r,r|imr"(i32 %tmp1) nounwind
  store i32 %1, i32* %out0, align 4
  %2 = call i32 asm "foo $1,$0", "=r|r,r|imr"(i32 1) nounwind
  store i32 %2, i32* %out0, align 4
  ret void
}

define void @multi_X() nounwind {
entry:
  %out0 = alloca i32, align 4
  %in1 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  store i32 1, i32* %in1, align 4
  %tmp = load i32, i32* %in1, align 4
  %0 = call i32 asm "foo $1,$0", "=r|r,r|X"(i32 %tmp) nounwind
  store i32 %0, i32* %out0, align 4
  %tmp1 = load i32, i32* @min1, align 4
  %1 = call i32 asm "foo $1,$0", "=r|r,r|X"(i32 %tmp1) nounwind
  store i32 %1, i32* %out0, align 4
  %2 = call i32 asm "foo $1,$0", "=r|r,r|X"(i32 1) nounwind
  store i32 %2, i32* %out0, align 4
  %3 = call i32 asm "foo $1,$0", "=r|r,r|X"(i32* getelementptr inbounds ([2 x i32], [2 x i32]* @marray, i32 0, i32 0)) nounwind
  store i32 %3, i32* %out0, align 4
; No lowering support.
;  %4 = call i32 asm "foo $1,$0", "=r|r,r|X"(double 1.000000e+001) nounwind
;  store i32 %4, i32* %out0, align 4
;  %5 = call i32 asm "foo $1,$0", "=r|r,r|X"(double 1.000000e+000) nounwind
;  store i32 %5, i32* %out0, align 4
  ret void
}

define void @multi_p() nounwind {
entry:
  %out0 = alloca i32, align 4
  store i32 0, i32* %out0, align 4
  %0 = call i32 asm "foo $1,$0", "=r|r,r|r"(i32* getelementptr inbounds ([2 x i32], [2 x i32]* @marray, i32 0, i32 0)) nounwind
  store i32 %0, i32* %out0, align 4
  ret void
}

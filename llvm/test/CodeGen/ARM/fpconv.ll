; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s --check-prefix=CHECK-VFP
; RUN: llc < %s -march=arm | FileCheck %s

define float @f1(double %x) {
;CHECK-VFP: f1:
;CHECK-VFP: fcvtsd
;CHECK: f1:
;CHECK: truncdfsf2
entry:
	%tmp1 = fptrunc double %x to float		; <float> [#uses=1]
	ret float %tmp1
}

define double @f2(float %x) {
;CHECK-VFP: f2:
;CHECK-VFP: fcvtds
;CHECK: f2:
;CHECK: extendsfdf2
entry:
	%tmp1 = fpext float %x to double		; <double> [#uses=1]
	ret double %tmp1
}

define i32 @f3(float %x) {
;CHECK-VFP: f3:
;CHECK-VFP: ftosizs
;CHECK: f3:
;CHECK: fixsfsi
entry:
	%tmp = fptosi float %x to i32		; <i32> [#uses=1]
	ret i32 %tmp
}

define i32 @f4(float %x) {
;CHECK-VFP: f4:
;CHECK-VFP: ftouizs
;CHECK: f4:
;CHECK: fixunssfsi
entry:
	%tmp = fptoui float %x to i32		; <i32> [#uses=1]
	ret i32 %tmp
}

define i32 @f5(double %x) {
;CHECK-VFP: f5:
;CHECK-VFP: ftosizd
;CHECK: f5:
;CHECK: fixdfsi
entry:
	%tmp = fptosi double %x to i32		; <i32> [#uses=1]
	ret i32 %tmp
}

define i32 @f6(double %x) {
;CHECK-VFP: f6:
;CHECK-VFP: ftouizd
;CHECK: f6:
;CHECK: fixunsdfsi
entry:
	%tmp = fptoui double %x to i32		; <i32> [#uses=1]
	ret i32 %tmp
}

define float @f7(i32 %a) {
;CHECK-VFP: f7:
;CHECK-VFP: fsitos
;CHECK: f7:
;CHECK: floatsisf
entry:
	%tmp = sitofp i32 %a to float		; <float> [#uses=1]
	ret float %tmp
}

define double @f8(i32 %a) {
;CHECK-VFP: f8:
;CHECK-VFP: fsitod
;CHECK: f8:
;CHECK: floatsidf
entry:
	%tmp = sitofp i32 %a to double		; <double> [#uses=1]
	ret double %tmp
}

define float @f9(i32 %a) {
;CHECK-VFP: f9:
;CHECK-VFP: fuitos
;CHECK: f9:
;CHECK: floatunsisf
entry:
	%tmp = uitofp i32 %a to float		; <float> [#uses=1]
	ret float %tmp
}

define double @f10(i32 %a) {
;CHECK-VFP: f10:
;CHECK-VFP: fuitod
;CHECK: f10:
;CHECK: floatunsidf
entry:
	%tmp = uitofp i32 %a to double		; <double> [#uses=1]
	ret double %tmp
}

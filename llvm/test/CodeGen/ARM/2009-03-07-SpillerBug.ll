; RUN: llvm-as < %s | llc -mtriple=armv6-apple-darwin9 -mattr=+vfp2
; rdar://6653182

	%struct.ggBRDF = type { i32 (...)** }
	%struct.ggPoint2 = type { [2 x double] }
	%struct.ggPoint3 = type { [3 x double] }
	%struct.ggSpectrum = type { [8 x float] }
	%struct.ggSphere = type { %struct.ggPoint3, double }
	%struct.mrDiffuseAreaSphereLuminaire = type { %struct.mrSphere, %struct.ggSpectrum }
	%struct.mrDiffuseCosineSphereLuminaire = type { %struct.mrDiffuseAreaSphereLuminaire }
	%struct.mrSphere = type { %struct.ggBRDF, %struct.ggSphere }

declare void @llvm.memcpy.i32(i8* nocapture, i8* nocapture, i32, i32) nounwind

declare double @llvm.sqrt.f64(double) nounwind readonly

declare double @sin(double) nounwind readonly

declare double @acos(double) nounwind readonly

define i32 @_ZNK34mrDiffuseSolidAngleSphereLuminaire18selectVisiblePointERK8ggPoint3RK9ggVector3RK8ggPoint2dRS0_Rd(%struct.mrDiffuseCosineSphereLuminaire* nocapture %this, %struct.ggPoint3* nocapture %x, %struct.ggPoint3* nocapture %unnamed_arg, %struct.ggPoint2* nocapture %uv, double %unnamed_arg2, %struct.ggPoint3* nocapture %on_light, double* nocapture %invProb) nounwind {
entry:
	%0 = call double @llvm.sqrt.f64(double 0.000000e+00) nounwind		; <double> [#uses=4]
	%1 = fcmp ult double 0.000000e+00, %0		; <i1> [#uses=1]
	br i1 %1, label %bb3, label %bb7

bb3:		; preds = %entry
	%2 = fdiv double 1.000000e+00, 0.000000e+00		; <double> [#uses=1]
	%3 = mul double 0.000000e+00, %2		; <double> [#uses=2]
	%4 = call double @llvm.sqrt.f64(double 0.000000e+00) nounwind		; <double> [#uses=1]
	%5 = fdiv double 1.000000e+00, %4		; <double> [#uses=2]
	%6 = mul double %3, %5		; <double> [#uses=2]
	%7 = mul double 0.000000e+00, %5		; <double> [#uses=2]
	%8 = mul double %3, %7		; <double> [#uses=1]
	%9 = sub double %8, 0.000000e+00		; <double> [#uses=1]
	%10 = mul double 0.000000e+00, %6		; <double> [#uses=1]
	%11 = sub double 0.000000e+00, %10		; <double> [#uses=1]
	%12 = sub double -0.000000e+00, %11		; <double> [#uses=1]
	%13 = mul double %0, %0		; <double> [#uses=2]
	%14 = sub double %13, 0.000000e+00		; <double> [#uses=1]
	%15 = call double @llvm.sqrt.f64(double %14)		; <double> [#uses=1]
	%16 = mul double 0.000000e+00, %15		; <double> [#uses=1]
	%17 = fdiv double %16, %0		; <double> [#uses=1]
	%18 = add double 0.000000e+00, %17		; <double> [#uses=1]
	%19 = call double @acos(double %18) nounwind readonly		; <double> [#uses=1]
	%20 = load double* null, align 4		; <double> [#uses=1]
	%21 = mul double %20, 0x401921FB54442D18		; <double> [#uses=1]
	%22 = call double @sin(double %19) nounwind readonly		; <double> [#uses=2]
	%23 = mul double %22, 0.000000e+00		; <double> [#uses=2]
	%24 = mul double %6, %23		; <double> [#uses=1]
	%25 = mul double %7, %23		; <double> [#uses=1]
	%26 = call double @sin(double %21) nounwind readonly		; <double> [#uses=1]
	%27 = mul double %22, %26		; <double> [#uses=2]
	%28 = mul double %9, %27		; <double> [#uses=1]
	%29 = mul double %27, %12		; <double> [#uses=1]
	%30 = add double %24, %28		; <double> [#uses=1]
	%31 = add double 0.000000e+00, %29		; <double> [#uses=1]
	%32 = add double %25, 0.000000e+00		; <double> [#uses=1]
	%33 = add double %30, 0.000000e+00		; <double> [#uses=1]
	%34 = add double %31, 0.000000e+00		; <double> [#uses=1]
	%35 = add double %32, 0.000000e+00		; <double> [#uses=1]
	%36 = bitcast %struct.ggPoint3* %x to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* null, i8* %36, i32 24, i32 4) nounwind
	store double %33, double* null, align 8
	br i1 false, label %_Z20ggRaySphereIntersectRK6ggRay3RK8ggSphereddRd.exit, label %bb5.i.i.i

bb5.i.i.i:		; preds = %bb3
	unreachable

_Z20ggRaySphereIntersectRK6ggRay3RK8ggSphereddRd.exit:		; preds = %bb3
	%37 = sub double %13, 0.000000e+00		; <double> [#uses=0]
	%38 = sub double -0.000000e+00, %34		; <double> [#uses=0]
	%39 = sub double -0.000000e+00, %35		; <double> [#uses=0]
	ret i32 1

bb7:		; preds = %entry
	ret i32 0
}

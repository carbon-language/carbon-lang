; RUN: llvm-as < %s -o - | lli -force-interpreter
; PR1629

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"
	%struct.X = type { i32 }
@_ZGVZ4mainE1a = internal global i64 0, align 8		; <i64*> [#uses=3]
@_ZZ4mainE1a = internal global %struct.X zeroinitializer		; <%struct.X*> [#uses=1]

define i32 @main() nounwind  {
entry:
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp2 = call double @sin( double 1.999000e+00 ) nounwind readonly 		; <double> [#uses=1]
	%tmp3 = call double @cos( double 1.990000e+00 ) nounwind readonly 		; <double> [#uses=1]
	%tmp4 = add double %tmp2, %tmp3		; <double> [#uses=1]
	%tmp5 = load i8* bitcast (i64* @_ZGVZ4mainE1a to i8*), align 1		; <i8> [#uses=1]
	%tmp6 = icmp eq i8 %tmp5, 0		; <i1> [#uses=1]
	%tmp67 = zext i1 %tmp6 to i8		; <i8> [#uses=1]
	%toBool = icmp ne i8 %tmp67, 0		; <i1> [#uses=1]
	br i1 %toBool, label %cond_true, label %cond_next14

cond_true:		; preds = %entry
	%tmp8 = call i32 @__cxa_guard_acquire( i64* @_ZGVZ4mainE1a ) nounwind 		; <i32> [#uses=1]
	%tmp9 = icmp ne i32 %tmp8, 0		; <i1> [#uses=1]
	%tmp910 = zext i1 %tmp9 to i8		; <i8> [#uses=1]
	%toBool12 = icmp ne i8 %tmp910, 0		; <i1> [#uses=1]
	br i1 %toBool12, label %cond_true13, label %cond_next14

cond_true13:		; preds = %cond_true
	call void @_ZN1XC1Ei( %struct.X* @_ZZ4mainE1a, i32 0 ) nounwind 
	call void @__cxa_guard_release( i64* @_ZGVZ4mainE1a ) nounwind 
	br label %cond_next14

cond_next14:		; preds = %cond_true13, %cond_true, %entry
	%tmp1516 = fptosi double %tmp4 to i32		; <i32> [#uses=1]
	ret i32 %tmp1516
}

define linkonce void @_ZN1XC1Ei(%struct.X* %this, i32 %val) nounwind  {
entry:
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp1 = getelementptr %struct.X* %this, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 %val, i32* %tmp1, align 4
	ret void
}

declare double @sin(double) nounwind readonly 

declare double @cos(double) nounwind readonly 

declare i32 @__cxa_guard_acquire(i64*) nounwind 

declare void @__cxa_guard_release(i64*) nounwind 

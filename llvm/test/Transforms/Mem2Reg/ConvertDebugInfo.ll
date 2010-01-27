; RUN: opt < %s -mem2reg -S | FileCheck %s

target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

define double @testfunc(i32 %i, double %j) {
	%I = alloca i32		; <i32*> [#uses=4]
  call void @llvm.dbg.declare(metadata !{i32* %I}, metadata !0)
	%J = alloca double		; <double*> [#uses=2]
  call void @llvm.dbg.declare(metadata !{double* %J}, metadata !1)
; CHECK: call void @llvm.dbg.value(metadata !{i32 %i}, i64 0, metadata !0)
	store i32 %i, i32* %I
; CHECK: call void @llvm.dbg.value(metadata !{double %j}, i64 0, metadata !1), !dbg !3
	store double %j, double* %J, !dbg !3
	%t1 = load i32* %I		; <i32> [#uses=1]
	%t2 = add i32 %t1, 1		; <i32> [#uses=1]
	store i32 %t2, i32* %I
	%t3 = load i32* %I		; <i32> [#uses=1]
	%t4 = sitofp i32 %t3 to double		; <double> [#uses=1]
	%t5 = load double* %J		; <double> [#uses=1]
	%t6 = fmul double %t4, %t5		; <double> [#uses=1]
	ret double %t6
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!bar = !{!0}
!foo = !{!2}

!0 = metadata !{i32 459008, metadata !1, metadata !"foo", metadata !2, i32 5, metadata !"foo"} ; [ DW_TAG_auto_variable ]
!1 = metadata !{i32 459008, metadata !1, metadata !"foo", metadata !0, i32 5, metadata !1} ; [ DW_TAG_auto_variable ]
!2 = metadata !{i32 458804, i32 0, metadata !2, metadata !"foo", metadata !"bar", metadata !"bar", metadata !2, i32 3, metadata !0, i1 false, i1 true} ; [ DW_TAG_variable ]
!3 = metadata !{i32 4, i32 0, metadata !0, null}

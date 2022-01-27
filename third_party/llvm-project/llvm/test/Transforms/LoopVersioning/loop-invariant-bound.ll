; RUN: opt -loop-versioning -S < %s | FileCheck %s
; Checks that when introducing check, we don't accidentally introduce non-dominating instructions
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%Dual.212 = type { %Dual.213, %Partials.215 }
%Dual.213 = type { double, %Partials.214 }
%Partials.214 = type { [2 x double] }
%Partials.215 = type { [2 x %Dual.213] }

; Function Attrs: sspreq
define void @"julia_axpy!_65480"(%Dual.212*, %Dual.212* %other) {
top:
  br label %if24

; CHECK-NOT: %bc = bitcast i64* %v2.sroa.0.0..sroa_cast
; CHECK: %bound0 = icmp ult i8* %[[x:[a-z0-9]+]], %[[y:[a-z0-9]+]]
; CHECK-NOT: %bound1 = icmp ult i8* %[[y]], %[[x]]

if24:                                             ; preds = %if24, %top
  %"#temp#1.sroa.3.02" = phi i64 [ undef, %top ], [ %2, %if24 ]
  %"#temp#1.sroa.0.01" = phi i64 [ undef, %top ], [ %1, %if24 ]
  %1 = add i64 %"#temp#1.sroa.0.01", 1
  %2 = add i64 %"#temp#1.sroa.3.02", 1
  ; This pointer is loop invariant. LAA used to re-use it from memcheck, even though it didn't dominate.
  %v2.sroa.0.0..sroa_cast = bitcast %Dual.212* %0 to i64*
  %v2.sroa.0.0.copyload = load i64, i64* %v2.sroa.0.0..sroa_cast, align 1
  %3 = add i64 %"#temp#1.sroa.0.01", -1
  %4 = getelementptr inbounds %Dual.212, %Dual.212* %other, i64 0, i32 1, i32 0, i64 0, i32 1, i32 0, i64 0
  %5 = bitcast double* %4 to i64*
  store i64 undef, i64* %5, align 8
  %notlhs27 = icmp eq i64 %2, undef
  %notrhs28 = icmp eq i64 %1, undef
  %6 = or i1 %notrhs28, %notlhs27
  br i1 %6, label %L41.L335_crit_edge, label %if24

L41.L335_crit_edge:                               ; preds = %if24
  ret void
}

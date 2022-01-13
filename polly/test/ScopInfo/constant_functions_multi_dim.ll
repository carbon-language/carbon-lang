; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-scops -analyze \
; RUN:                -polly-detect-full-functions < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK:         Statements {
; CHECK-NEXT:    	Stmt_entry_split
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_entry_split[] };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_entry_split[] -> [0, 0] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_entry_split[] -> MemRef_acc_0_lcssa__phi[] };
; CHECK-NEXT:    	Stmt_for_inc_lr_ph
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_inc_lr_ph[] : N > 0 };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_inc_lr_ph[] -> [1, 0] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_inc_lr_ph[] -> MemRef_acc_03__phi[] };
; CHECK-NEXT:    	Stmt_for_inc
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_inc[i0] : 0 <= i0 < N };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_inc[i0] -> [2, i0] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_inc[i0] -> MemRef_acc_03__phi[] };
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_inc[i0] -> MemRef_acc_03__phi[] };
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_inc[i0] -> MemRef_A[__global_id_0, i0] };
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_inc[i0] -> MemRef_B[i0, __global_id_1] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_inc[i0] -> MemRef__lcssa__phi[] };
; CHECK-NEXT:    	Stmt_for_cond_for_end_crit_edge
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_cond_for_end_crit_edge[] : N > 0 };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_cond_for_end_crit_edge[] -> [3, 0] };
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_cond_for_end_crit_edge[] -> MemRef__lcssa__phi[] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_cond_for_end_crit_edge[] -> MemRef_acc_0_lcssa__phi[] };
; CHECK-NEXT:    	Stmt_for_end
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_end[] };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_end[] -> [4, 0] };
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_end[] -> MemRef_acc_0_lcssa__phi[] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [N, __global_id_0, __global_id_1] -> { Stmt_for_end[] -> MemRef_C[__global_id_0, __global_id_1] };
; CHECK-NEXT:    }


; Function Attrs: noinline nounwind uwtable
define void @mat_mul(float* %C, float* %A, float* %B, i64 %N) #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %call = tail call i64 @_Z13get_global_idj(i32 0) #3
  %call1 = tail call i64 @_Z13get_global_idj(i32 1) #3
  %cmp1 = icmp sgt i64 %N, 0
  %mul = mul nsw i64 %call, %N
  br i1 %cmp1, label %for.inc.lr.ph, label %for.end

for.inc.lr.ph:                                    ; preds = %entry.split
  br label %for.inc

for.inc:                                          ; preds = %for.inc.lr.ph, %for.inc
  %acc.03 = phi float [ 0.000000e+00, %for.inc.lr.ph ], [ %tmp6, %for.inc ]
  %m.02 = phi i64 [ 0, %for.inc.lr.ph ], [ %inc, %for.inc ]
  %add = add nsw i64 %m.02, %mul
  %arrayidx = getelementptr inbounds float, float* %A, i64 %add
  %tmp = load float, float* %arrayidx, align 4
  %mul2 = mul nsw i64 %m.02, %N
  %add3 = add nsw i64 %mul2, %call1
  %arrayidx4 = getelementptr inbounds float, float* %B, i64 %add3
  %tmp5 = load float, float* %arrayidx4, align 4
  %tmp6 = tail call float @llvm.fmuladd.f32(float %tmp, float %tmp5, float %acc.03)
  %inc = add nuw nsw i64 %m.02, 1
  %exitcond = icmp ne i64 %inc, %N
  br i1 %exitcond, label %for.inc, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.inc
  %.lcssa = phi float [ %tmp6, %for.inc ]
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry.split
  %acc.0.lcssa = phi float [ %.lcssa, %for.cond.for.end_crit_edge ], [ 0.000000e+00, %entry.split ]
  %add7 = add nsw i64 %mul, %call1
  %arrayidx8 = getelementptr inbounds float, float* %C, i64 %add7
  store float %acc.0.lcssa, float* %arrayidx8, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare i64 @_Z13get_global_idj(i32) #1

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fmuladd.f32(float, float, float) #2

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind readnone }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (trunk 303846) (llvm/trunk 303834)"}
!2 = !{i32 1, i32 1, i32 1, i32 0}
!3 = !{!"none", !"none", !"none", !"none"}
!4 = !{!"float*", !"float*", !"float*", !"long"}
!5 = !{!"", !"", !"", !""}

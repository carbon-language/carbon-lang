; RUN: opt %loadPolly -polly-print-scops -polly-invariant-load-hoisting=true -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -polly-invariant-load-hoisting=true -disable-output < %s

; CHECK:      Statements {
; CHECK-NEXT: 	Stmt_L_4
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [tmp8, tmp22, tmp15] -> { Stmt_L_4[i0, i1, i2] : 0 <= i0 < tmp8 and 0 <= i1 < tmp8 and 0 <= i2 < tmp8 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [tmp8, tmp22, tmp15] -> { Stmt_L_4[i0, i1, i2] -> [i0, i1, i2] };
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [tmp8, tmp22, tmp15] -> { Stmt_L_4[i0, i1, i2] -> MemRef_tmp19[i1, i0] };
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [tmp8, tmp22, tmp15] -> { Stmt_L_4[i0, i1, i2] -> MemRef_tmp5[i2, i0] };
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [tmp8, tmp22, tmp15] -> { Stmt_L_4[i0, i1, i2] -> MemRef_tmp12[i2, i1] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [tmp8, tmp22, tmp15] -> { Stmt_L_4[i0, i1, i2] -> MemRef_tmp19[i1, i0] };
; CHECK-NEXT: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%jl_value_t = type { %jl_value_t* }

define %jl_value_t* @julia_gemm_22583(%jl_value_t*, %jl_value_t** %tmp1, i32) {
top:
  br label %top.split

top.split:                                        ; preds = %top
  %tmp3 = load %jl_value_t*, %jl_value_t** %tmp1, align 8
  %tmp4 = bitcast %jl_value_t* %tmp3 to double**
  %tmp5 = load double*, double** %tmp4, align 8
  %tmp6 = getelementptr inbounds %jl_value_t, %jl_value_t* %tmp3, i64 3, i32 0
  %tmp7 = bitcast %jl_value_t** %tmp6 to i64*
  %tmp8 = load i64, i64* %tmp7, align 8
  %tmp9 = getelementptr %jl_value_t*, %jl_value_t** %tmp1, i64 1
  %tmp10 = load %jl_value_t*, %jl_value_t** %tmp9, align 8
  %tmp11 = bitcast %jl_value_t* %tmp10 to double**
  %tmp12 = load double*, double** %tmp11, align 8
  %tmp13 = getelementptr inbounds %jl_value_t, %jl_value_t* %tmp10, i64 3, i32 0
  %tmp14 = bitcast %jl_value_t** %tmp13 to i64*
  %tmp15 = load i64, i64* %tmp14, align 8
  %tmp16 = getelementptr %jl_value_t*, %jl_value_t** %tmp1, i64 2
  %tmp17 = load %jl_value_t*, %jl_value_t** %tmp16, align 8
  %tmp18 = bitcast %jl_value_t* %tmp17 to double**
  %tmp19 = load double*, double** %tmp18, align 8
  %tmp20 = getelementptr inbounds %jl_value_t, %jl_value_t* %tmp17, i64 3, i32 0
  %tmp21 = bitcast %jl_value_t** %tmp20 to i64*
  %tmp22 = load i64, i64* %tmp21, align 8
  %tmp23 = icmp sgt i64 %tmp8, 0
  %tmp24 = select i1 %tmp23, i64 %tmp8, i64 0
  %tmp25 = add i64 %tmp24, 1
  %tmp26 = icmp eq i64 %tmp24, 0
  br i1 %tmp26, label %L.11, label %L.preheader

L.preheader:                                      ; preds = %top.split
  br label %L

L:                                                ; preds = %L.preheader, %L.9
  %"#s5.0" = phi i64 [ %tmp27, %L.9 ], [ 1, %L.preheader ]
  %tmp27 = add i64 %"#s5.0", 1
  br i1 %tmp26, label %L.9, label %L.2.preheader

L.2.preheader:                                    ; preds = %L
  br label %L.2

L.2:                                              ; preds = %L.2.preheader, %L.7
  %"#s4.0" = phi i64 [ %tmp28, %L.7 ], [ 1, %L.2.preheader ]
  %tmp28 = add i64 %"#s4.0", 1
  br i1 %tmp26, label %L.7, label %L.4.preheader

L.4.preheader:                                    ; preds = %L.2
  br label %L.4

L.4:                                              ; preds = %L.4.preheader, %L.4
  %"#s3.0" = phi i64 [ %tmp29, %L.4 ], [ 1, %L.4.preheader ]
  %tmp29 = add i64 %"#s3.0", 1
  %tmp30 = add i64 %"#s5.0", -1
  %tmp31 = add i64 %"#s4.0", -1
  %tmp32 = mul i64 %tmp31, %tmp22
  %tmp33 = add i64 %tmp32, %tmp30
  %tmp34 = getelementptr double, double* %tmp19, i64 %tmp33
  %tmp35 = load double, double* %tmp34, align 8
  %tmp36 = add i64 %"#s3.0", -1
  %tmp37 = mul i64 %tmp36, %tmp8
  %tmp38 = add i64 %tmp37, %tmp30
  %tmp39 = getelementptr double, double* %tmp5, i64 %tmp38
  %tmp40 = load double, double* %tmp39, align 8
  %tmp41 = mul i64 %tmp36, %tmp15
  %tmp42 = add i64 %tmp41, %tmp31
  %tmp43 = getelementptr double, double* %tmp12, i64 %tmp42
  %tmp44 = load double, double* %tmp43, align 8
  %tmp45 = fmul double %tmp40, %tmp44
  %tmp46 = fadd double %tmp35, %tmp45
  store double %tmp46, double* %tmp34, align 8
  %tmp47 = icmp eq i64 %tmp29, %tmp25
  br i1 %tmp47, label %L.7.loopexit, label %L.4

L.7.loopexit:                                     ; preds = %L.4
  br label %L.7

L.7:                                              ; preds = %L.7.loopexit, %L.2
  %tmp48 = icmp eq i64 %tmp28, %tmp25
  br i1 %tmp48, label %L.9.loopexit, label %L.2

L.9.loopexit:                                     ; preds = %L.7
  br label %L.9

L.9:                                              ; preds = %L.9.loopexit, %L
  %tmp49 = icmp eq i64 %tmp27, %tmp25
  br i1 %tmp49, label %L.11.loopexit, label %L

L.11.loopexit:                                    ; preds = %L.9
  br label %L.11

L.11:                                             ; preds = %L.11.loopexit, %top.split
  ret %jl_value_t* inttoptr (i64 140220477440016 to %jl_value_t*)
}

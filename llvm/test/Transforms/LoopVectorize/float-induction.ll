; RUN: opt < %s  -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine -S | FileCheck --check-prefix VEC4_INTERL1 %s
; RUN: opt < %s  -loop-vectorize -force-vector-interleave=2 -force-vector-width=4 -dce -instcombine -S | FileCheck --check-prefix VEC4_INTERL2 %s
; RUN: opt < %s  -loop-vectorize -force-vector-interleave=2 -force-vector-width=1 -dce -instcombine -S | FileCheck --check-prefix VEC1_INTERL2 %s
; RUN: opt < %s  -loop-vectorize -force-vector-interleave=1 -force-vector-width=2 -dce -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -instcombine -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -keep-loops=false -S | FileCheck --check-prefix VEC2_INTERL1_PRED_STORE %s

@fp_inc = common global float 0.000000e+00, align 4

;void fp_iv_loop1(float init, float * __restrict__ A, int N) {
;  float x = init;
;  for (int i=0; i < N; ++i) {
;    A[i] = x;
;    x -= fp_inc;
;  }
;}

; VEC4_INTERL1-LABEL: @fp_iv_loop1_fast_FMF(
; VEC4_INTERL1:       vector.ph:
; VEC4_INTERL1:         [[DOTSPLATINSERT:%.*]] = insertelement <4 x float> poison, float %init, i32 0
; VEC4_INTERL1-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <4 x float> [[DOTSPLATINSERT]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL1-NEXT:    [[DOTSPLATINSERT2:%.*]] = insertelement <4 x float> poison, float %fpinc, i32 0
; VEC4_INTERL1-NEXT:    [[DOTSPLAT3:%.*]] = shufflevector <4 x float> [[DOTSPLATINSERT2]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL1-NEXT:    [[TMP5:%.*]] = fmul fast <4 x float> [[DOTSPLAT3]], <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>
; VEC4_INTERL1-NEXT:    [[INDUCTION4:%.*]] = fsub fast <4 x float> [[DOTSPLAT]], [[TMP5]]
; VEC4_INTERL1-NEXT:    [[TMP6:%.*]] = fmul fast float %fpinc, 4.000000e+00
; VEC4_INTERL1-NEXT:    [[DOTSPLATINSERT5:%.*]] = insertelement <4 x float> poison, float [[TMP6]], i32 0
; VEC4_INTERL1-NEXT:    [[DOTSPLAT6:%.*]] = shufflevector <4 x float> [[DOTSPLATINSERT5]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL1-NEXT:    br label %vector.body
; VEC4_INTERL1:       vector.body:
; VEC4_INTERL1-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; VEC4_INTERL1-NEXT:    [[VEC_IND:%.*]] = phi <4 x float> [ [[INDUCTION4]], %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %vector.body ]
; VEC4_INTERL1-NEXT:    [[TMP8:%.*]] = getelementptr inbounds float, float* %A, i64 [[INDEX]]
; VEC4_INTERL1-NEXT:    [[TMP9:%.*]] = bitcast float* [[TMP8]] to <4 x float>*
; VEC4_INTERL1-NEXT:    store <4 x float> [[VEC_IND]], <4 x float>* [[TMP9]], align 4
; VEC4_INTERL1-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; VEC4_INTERL1-NEXT:    [[VEC_IND_NEXT]] = fsub fast <4 x float> [[VEC_IND]], [[DOTSPLAT6]]
; VEC4_INTERL1:         br i1 {{.*}}, label %middle.block, label %vector.body

; VEC4_INTERL2-LABEL: @fp_iv_loop1_fast_FMF(
; VEC4_INTERL2:       vector.ph:
; VEC4_INTERL2:         [[DOTSPLATINSERT:%.*]] = insertelement <4 x float> poison, float %init, i32 0
; VEC4_INTERL2-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <4 x float> [[DOTSPLATINSERT]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL2-NEXT:    [[DOTSPLATINSERT3:%.*]] = insertelement <4 x float> poison, float %fpinc, i32 0
; VEC4_INTERL2-NEXT:    [[DOTSPLAT4:%.*]] = shufflevector <4 x float> [[DOTSPLATINSERT3]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL2-NEXT:    [[TMP5:%.*]] = fmul fast <4 x float> [[DOTSPLAT4]], <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>
; VEC4_INTERL2-NEXT:    [[INDUCTION5:%.*]] = fsub fast <4 x float> [[DOTSPLAT]], [[TMP5]]
; VEC4_INTERL2-NEXT:    [[TMP6:%.*]] = fmul fast float %fpinc, 4.000000e+00
; VEC4_INTERL2-NEXT:    [[DOTSPLATINSERT6:%.*]] = insertelement <4 x float> poison, float [[TMP6]], i32 0
; VEC4_INTERL2-NEXT:    [[DOTSPLAT7:%.*]] = shufflevector <4 x float> [[DOTSPLATINSERT6]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL2-NEXT:    br label %vector.body
; VEC4_INTERL2:       vector.body:
; VEC4_INTERL2-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; VEC4_INTERL2-NEXT:    [[VEC_IND:%.*]] = phi <4 x float> [ [[INDUCTION5]], %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %vector.body ]
; VEC4_INTERL2-NEXT:    [[STEP_ADD:%.*]] = fsub fast <4 x float> [[VEC_IND]], [[DOTSPLAT7]]
; VEC4_INTERL2-NEXT:    [[TMP9:%.*]] = getelementptr inbounds float, float* %A, i64 [[INDEX]]
; VEC4_INTERL2-NEXT:    [[TMP10:%.*]] = bitcast float* [[TMP9]] to <4 x float>*
; VEC4_INTERL2-NEXT:    store <4 x float> [[VEC_IND]], <4 x float>* [[TMP10]], align 4
; VEC4_INTERL2-NEXT:    [[TMP11:%.*]] = getelementptr inbounds float, float* [[TMP9]], i64 4
; VEC4_INTERL2-NEXT:    [[TMP12:%.*]] = bitcast float* [[TMP11]] to <4 x float>*
; VEC4_INTERL2-NEXT:    store <4 x float> [[STEP_ADD]], <4 x float>* [[TMP12]], align 4
; VEC4_INTERL2-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 8
; VEC4_INTERL2-NEXT:    [[VEC_IND_NEXT]] = fsub fast <4 x float> [[STEP_ADD]], [[DOTSPLAT7]]
; VEC4_INTERL2:         br i1 {{.*}}, label %middle.block, label %vector.body

; VEC1_INTERL2-LABEL: @fp_iv_loop1_fast_FMF(
; VEC1_INTERL2:       vector.ph:
; VEC1_INTERL2:         br label %vector.body
; VEC1_INTERL2:       vector.body:
; VEC1_INTERL2-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; VEC1_INTERL2-NEXT:    [[INDUCTION2:%.*]] = or i64 [[INDEX]], 1
; VEC1_INTERL2-NEXT:    [[TMP6:%.*]] = sitofp i64 [[INDEX]] to float
; VEC1_INTERL2-NEXT:    [[TMP7:%.*]] = fmul fast float %fpinc, [[TMP6]]
; VEC1_INTERL2-NEXT:    [[FP_OFFSET_IDX:%.*]] = fsub fast float %init, [[TMP7]]
; VEC1_INTERL2-NEXT:    [[TMP8:%.*]] = fsub fast float [[FP_OFFSET_IDX]], %fpinc
; VEC1_INTERL2-NEXT:    [[TMP9:%.*]] = getelementptr inbounds float, float* %A, i64 [[INDEX]]
; VEC1_INTERL2-NEXT:    [[TMP10:%.*]] = getelementptr inbounds float, float* %A, i64 [[INDUCTION2]]
; VEC1_INTERL2-NEXT:    store float [[FP_OFFSET_IDX]], float* [[TMP9]], align 4
; VEC1_INTERL2-NEXT:    store float [[TMP8]], float* [[TMP10]], align 4
; VEC1_INTERL2-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; VEC1_INTERL2:         br i1 {{.*}}, label %middle.block, label %vector.body

define void @fp_iv_loop1_fast_FMF(float %init, float* noalias nocapture %A, i32 %N) {
entry:
  %cmp4 = icmp sgt i32 %N, 0
  br i1 %cmp4, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %fpinc = load float, float* @fp_inc, align 4
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %x.05 = phi float [ %init, %for.body.lr.ph ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %A, i64 %indvars.iv
  store float %x.05, float* %arrayidx, align 4
  %add = fsub fast float %x.05, %fpinc
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

; We do not need the full 'fast' FMF to vectorize the loop, but the code can't become
; 'fast' spontaneously - FMF should propagate from the original IR.

define void @fp_iv_loop1_reassoc_FMF(float %init, float* noalias nocapture %A, i32 %N) {
; VEC4_INTERL1-LABEL: @fp_iv_loop1_reassoc_FMF(
; VEC4_INTERL1-NEXT:  entry:
; VEC4_INTERL1-NEXT:    [[CMP4:%.*]] = icmp sgt i32 [[N:%.*]], 0
; VEC4_INTERL1-NEXT:    br i1 [[CMP4]], label [[FOR_BODY_LR_PH:%.*]], label [[FOR_END:%.*]]
; VEC4_INTERL1:       for.body.lr.ph:
; VEC4_INTERL1-NEXT:    [[FPINC:%.*]] = load float, float* @fp_inc, align 4
; VEC4_INTERL1-NEXT:    [[TMP0:%.*]] = add i32 [[N]], -1
; VEC4_INTERL1-NEXT:    [[TMP1:%.*]] = zext i32 [[TMP0]] to i64
; VEC4_INTERL1-NEXT:    [[TMP2:%.*]] = add nuw nsw i64 [[TMP1]], 1
; VEC4_INTERL1-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i32 [[TMP0]], 3
; VEC4_INTERL1-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; VEC4_INTERL1:       vector.ph:
; VEC4_INTERL1-NEXT:    [[N_VEC:%.*]] = and i64 [[TMP2]], 8589934588
; VEC4_INTERL1-NEXT:    [[CAST_CRD:%.*]] = sitofp i64 [[N_VEC]] to float
; VEC4_INTERL1-NEXT:    [[TMP3:%.*]] = fmul reassoc float [[FPINC]], [[CAST_CRD]]
; VEC4_INTERL1-NEXT:    [[IND_END:%.*]] = fsub reassoc float [[INIT:%.*]], [[TMP3]]
; VEC4_INTERL1-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <4 x float> poison, float [[INIT]], i32 0
; VEC4_INTERL1-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <4 x float> [[DOTSPLATINSERT]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL1-NEXT:    [[DOTSPLATINSERT2:%.*]] = insertelement <4 x float> poison, float [[FPINC]], i32 0
; VEC4_INTERL1-NEXT:    [[DOTSPLAT3:%.*]] = shufflevector <4 x float> [[DOTSPLATINSERT2]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL1-NEXT:    [[TMP4:%.*]] = fmul reassoc <4 x float> [[DOTSPLAT3]], <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>
; VEC4_INTERL1-NEXT:    [[INDUCTION:%.*]] = fsub reassoc <4 x float> [[DOTSPLAT]], [[TMP4]]
; VEC4_INTERL1-NEXT:    [[TMP5:%.*]] = fmul reassoc float [[FPINC]], 4.000000e+00
; VEC4_INTERL1-NEXT:    [[DOTSPLATINSERT4:%.*]] = insertelement <4 x float> poison, float [[TMP5]], i32 0
; VEC4_INTERL1-NEXT:    [[DOTSPLAT5:%.*]] = shufflevector <4 x float> [[DOTSPLATINSERT4]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL1-NEXT:    br label [[VECTOR_BODY:%.*]]
; VEC4_INTERL1:       vector.body:
; VEC4_INTERL1-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; VEC4_INTERL1-NEXT:    [[VEC_IND:%.*]] = phi <4 x float> [ [[INDUCTION]], [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[VECTOR_BODY]] ]
; VEC4_INTERL1-NEXT:    [[TMP6:%.*]] = getelementptr inbounds float, float* [[A:%.*]], i64 [[INDEX]]
; VEC4_INTERL1-NEXT:    [[TMP7:%.*]] = bitcast float* [[TMP6]] to <4 x float>*
; VEC4_INTERL1-NEXT:    store <4 x float> [[VEC_IND]], <4 x float>* [[TMP7]], align 4
; VEC4_INTERL1-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; VEC4_INTERL1-NEXT:    [[VEC_IND_NEXT]] = fsub reassoc <4 x float> [[VEC_IND]], [[DOTSPLAT5]]
; VEC4_INTERL1-NEXT:    [[TMP8:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; VEC4_INTERL1-NEXT:    br i1 [[TMP8]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], [[LOOP4:!llvm.loop !.*]]
; VEC4_INTERL1:       middle.block:
; VEC4_INTERL1-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[TMP2]], [[N_VEC]]
; VEC4_INTERL1-NEXT:    br i1 [[CMP_N]], label [[FOR_END_LOOPEXIT:%.*]], label [[SCALAR_PH]]
; VEC4_INTERL1:       scalar.ph:
; VEC4_INTERL1-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[FOR_BODY_LR_PH]] ]
; VEC4_INTERL1-NEXT:    [[BC_RESUME_VAL1:%.*]] = phi float [ [[IND_END]], [[MIDDLE_BLOCK]] ], [ [[INIT]], [[FOR_BODY_LR_PH]] ]
; VEC4_INTERL1-NEXT:    br label [[FOR_BODY:%.*]]
; VEC4_INTERL1:       for.body:
; VEC4_INTERL1-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ], [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ]
; VEC4_INTERL1-NEXT:    [[X_05:%.*]] = phi float [ [[BC_RESUME_VAL1]], [[SCALAR_PH]] ], [ [[ADD:%.*]], [[FOR_BODY]] ]
; VEC4_INTERL1-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[INDVARS_IV]]
; VEC4_INTERL1-NEXT:    store float [[X_05]], float* [[ARRAYIDX]], align 4
; VEC4_INTERL1-NEXT:    [[ADD]] = fsub reassoc float [[X_05]], [[FPINC]]
; VEC4_INTERL1-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; VEC4_INTERL1-NEXT:    [[LFTR_WIDEIV:%.*]] = trunc i64 [[INDVARS_IV_NEXT]] to i32
; VEC4_INTERL1-NEXT:    [[EXITCOND:%.*]] = icmp eq i32 [[LFTR_WIDEIV]], [[N]]
; VEC4_INTERL1-NEXT:    br i1 [[EXITCOND]], label [[FOR_END_LOOPEXIT]], label [[FOR_BODY]], [[LOOP5:!llvm.loop !.*]]
; VEC4_INTERL1:       for.end.loopexit:
; VEC4_INTERL1-NEXT:    br label [[FOR_END]]
; VEC4_INTERL1:       for.end:
; VEC4_INTERL1-NEXT:    ret void
;
; VEC4_INTERL2-LABEL: @fp_iv_loop1_reassoc_FMF(
; VEC4_INTERL2-NEXT:  entry:
; VEC4_INTERL2-NEXT:    [[CMP4:%.*]] = icmp sgt i32 [[N:%.*]], 0
; VEC4_INTERL2-NEXT:    br i1 [[CMP4]], label [[FOR_BODY_LR_PH:%.*]], label [[FOR_END:%.*]]
; VEC4_INTERL2:       for.body.lr.ph:
; VEC4_INTERL2-NEXT:    [[FPINC:%.*]] = load float, float* @fp_inc, align 4
; VEC4_INTERL2-NEXT:    [[TMP0:%.*]] = add i32 [[N]], -1
; VEC4_INTERL2-NEXT:    [[TMP1:%.*]] = zext i32 [[TMP0]] to i64
; VEC4_INTERL2-NEXT:    [[TMP2:%.*]] = add nuw nsw i64 [[TMP1]], 1
; VEC4_INTERL2-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i32 [[TMP0]], 7
; VEC4_INTERL2-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; VEC4_INTERL2:       vector.ph:
; VEC4_INTERL2-NEXT:    [[N_VEC:%.*]] = and i64 [[TMP2]], 8589934584
; VEC4_INTERL2-NEXT:    [[CAST_CRD:%.*]] = sitofp i64 [[N_VEC]] to float
; VEC4_INTERL2-NEXT:    [[TMP3:%.*]] = fmul reassoc float [[FPINC]], [[CAST_CRD]]
; VEC4_INTERL2-NEXT:    [[IND_END:%.*]] = fsub reassoc float [[INIT:%.*]], [[TMP3]]
; VEC4_INTERL2-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <4 x float> poison, float [[INIT]], i32 0
; VEC4_INTERL2-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <4 x float> [[DOTSPLATINSERT]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL2-NEXT:    [[DOTSPLATINSERT2:%.*]] = insertelement <4 x float> poison, float [[FPINC]], i32 0
; VEC4_INTERL2-NEXT:    [[DOTSPLAT3:%.*]] = shufflevector <4 x float> [[DOTSPLATINSERT2]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL2-NEXT:    [[TMP4:%.*]] = fmul reassoc <4 x float> [[DOTSPLAT3]], <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>
; VEC4_INTERL2-NEXT:    [[INDUCTION:%.*]] = fsub reassoc <4 x float> [[DOTSPLAT]], [[TMP4]]
; VEC4_INTERL2-NEXT:    [[TMP5:%.*]] = fmul reassoc float [[FPINC]], 4.000000e+00
; VEC4_INTERL2-NEXT:    [[DOTSPLATINSERT4:%.*]] = insertelement <4 x float> poison, float [[TMP5]], i32 0
; VEC4_INTERL2-NEXT:    [[DOTSPLAT5:%.*]] = shufflevector <4 x float> [[DOTSPLATINSERT4]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL2-NEXT:    br label [[VECTOR_BODY:%.*]]
; VEC4_INTERL2:       vector.body:
; VEC4_INTERL2-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; VEC4_INTERL2-NEXT:    [[VEC_IND:%.*]] = phi <4 x float> [ [[INDUCTION]], [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[VECTOR_BODY]] ]
; VEC4_INTERL2-NEXT:    [[STEP_ADD:%.*]] = fsub reassoc <4 x float> [[VEC_IND]], [[DOTSPLAT5]]
; VEC4_INTERL2-NEXT:    [[TMP6:%.*]] = getelementptr inbounds float, float* [[A:%.*]], i64 [[INDEX]]
; VEC4_INTERL2-NEXT:    [[TMP7:%.*]] = bitcast float* [[TMP6]] to <4 x float>*
; VEC4_INTERL2-NEXT:    store <4 x float> [[VEC_IND]], <4 x float>* [[TMP7]], align 4
; VEC4_INTERL2-NEXT:    [[TMP8:%.*]] = getelementptr inbounds float, float* [[TMP6]], i64 4
; VEC4_INTERL2-NEXT:    [[TMP9:%.*]] = bitcast float* [[TMP8]] to <4 x float>*
; VEC4_INTERL2-NEXT:    store <4 x float> [[STEP_ADD]], <4 x float>* [[TMP9]], align 4
; VEC4_INTERL2-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 8
; VEC4_INTERL2-NEXT:    [[VEC_IND_NEXT]] = fsub reassoc <4 x float> [[STEP_ADD]], [[DOTSPLAT5]]
; VEC4_INTERL2-NEXT:    [[TMP10:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; VEC4_INTERL2-NEXT:    br i1 [[TMP10]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], [[LOOP4:!llvm.loop !.*]]
; VEC4_INTERL2:       middle.block:
; VEC4_INTERL2-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[TMP2]], [[N_VEC]]
; VEC4_INTERL2-NEXT:    br i1 [[CMP_N]], label [[FOR_END_LOOPEXIT:%.*]], label [[SCALAR_PH]]
; VEC4_INTERL2:       scalar.ph:
; VEC4_INTERL2-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[FOR_BODY_LR_PH]] ]
; VEC4_INTERL2-NEXT:    [[BC_RESUME_VAL1:%.*]] = phi float [ [[IND_END]], [[MIDDLE_BLOCK]] ], [ [[INIT]], [[FOR_BODY_LR_PH]] ]
; VEC4_INTERL2-NEXT:    br label [[FOR_BODY:%.*]]
; VEC4_INTERL2:       for.body:
; VEC4_INTERL2-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ], [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ]
; VEC4_INTERL2-NEXT:    [[X_05:%.*]] = phi float [ [[BC_RESUME_VAL1]], [[SCALAR_PH]] ], [ [[ADD:%.*]], [[FOR_BODY]] ]
; VEC4_INTERL2-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[INDVARS_IV]]
; VEC4_INTERL2-NEXT:    store float [[X_05]], float* [[ARRAYIDX]], align 4
; VEC4_INTERL2-NEXT:    [[ADD]] = fsub reassoc float [[X_05]], [[FPINC]]
; VEC4_INTERL2-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; VEC4_INTERL2-NEXT:    [[LFTR_WIDEIV:%.*]] = trunc i64 [[INDVARS_IV_NEXT]] to i32
; VEC4_INTERL2-NEXT:    [[EXITCOND:%.*]] = icmp eq i32 [[LFTR_WIDEIV]], [[N]]
; VEC4_INTERL2-NEXT:    br i1 [[EXITCOND]], label [[FOR_END_LOOPEXIT]], label [[FOR_BODY]], [[LOOP5:!llvm.loop !.*]]
; VEC4_INTERL2:       for.end.loopexit:
; VEC4_INTERL2-NEXT:    br label [[FOR_END]]
; VEC4_INTERL2:       for.end:
; VEC4_INTERL2-NEXT:    ret void
;
; VEC1_INTERL2-LABEL: @fp_iv_loop1_reassoc_FMF(
; VEC1_INTERL2-NEXT:  entry:
; VEC1_INTERL2-NEXT:    [[CMP4:%.*]] = icmp sgt i32 [[N:%.*]], 0
; VEC1_INTERL2-NEXT:    br i1 [[CMP4]], label [[FOR_BODY_LR_PH:%.*]], label [[FOR_END:%.*]]
; VEC1_INTERL2:       for.body.lr.ph:
; VEC1_INTERL2-NEXT:    [[FPINC:%.*]] = load float, float* @fp_inc, align 4
; VEC1_INTERL2-NEXT:    [[TMP0:%.*]] = add i32 [[N]], -1
; VEC1_INTERL2-NEXT:    [[TMP1:%.*]] = zext i32 [[TMP0]] to i64
; VEC1_INTERL2-NEXT:    [[TMP2:%.*]] = add nuw nsw i64 [[TMP1]], 1
; VEC1_INTERL2-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp eq i32 [[TMP0]], 0
; VEC1_INTERL2-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; VEC1_INTERL2:       vector.ph:
; VEC1_INTERL2-NEXT:    [[N_VEC:%.*]] = and i64 [[TMP2]], 8589934590
; VEC1_INTERL2-NEXT:    [[CAST_CRD:%.*]] = sitofp i64 [[N_VEC]] to float
; VEC1_INTERL2-NEXT:    [[TMP3:%.*]] = fmul reassoc float [[FPINC]], [[CAST_CRD]]
; VEC1_INTERL2-NEXT:    [[IND_END:%.*]] = fsub reassoc float [[INIT:%.*]], [[TMP3]]
; VEC1_INTERL2-NEXT:    br label [[VECTOR_BODY:%.*]]
; VEC1_INTERL2:       vector.body:
; VEC1_INTERL2-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; VEC1_INTERL2-NEXT:    [[INDUCTION2:%.*]] = or i64 [[INDEX]], 1
; VEC1_INTERL2-NEXT:    [[TMP4:%.*]] = sitofp i64 [[INDEX]] to float
; VEC1_INTERL2-NEXT:    [[TMP5:%.*]] = fmul reassoc float [[FPINC]], [[TMP4]]
; VEC1_INTERL2-NEXT:    [[OFFSET_IDX:%.*]] = fsub reassoc float [[INIT]], [[TMP5]]
; VEC1_INTERL2-NEXT:    [[TMP6:%.*]] = fmul reassoc float [[FPINC]], 0.000000e+00
; VEC1_INTERL2-NEXT:    [[TMP7:%.*]] = fsub reassoc float [[OFFSET_IDX]], [[TMP6]]
; VEC1_INTERL2-NEXT:    [[TMP8:%.*]] = fsub reassoc float [[OFFSET_IDX]], [[FPINC]]
; VEC1_INTERL2-NEXT:    [[TMP9:%.*]] = getelementptr inbounds float, float* [[A:%.*]], i64 [[INDEX]]
; VEC1_INTERL2-NEXT:    [[TMP10:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[INDUCTION2]]
; VEC1_INTERL2-NEXT:    store float [[TMP7]], float* [[TMP9]], align 4
; VEC1_INTERL2-NEXT:    store float [[TMP8]], float* [[TMP10]], align 4
; VEC1_INTERL2-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; VEC1_INTERL2-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; VEC1_INTERL2-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; VEC1_INTERL2:       middle.block:
; VEC1_INTERL2-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[TMP2]], [[N_VEC]]
; VEC1_INTERL2-NEXT:    br i1 [[CMP_N]], label [[FOR_END_LOOPEXIT:%.*]], label [[SCALAR_PH]]
; VEC1_INTERL2:       scalar.ph:
; VEC1_INTERL2-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[FOR_BODY_LR_PH]] ]
; VEC1_INTERL2-NEXT:    [[BC_RESUME_VAL1:%.*]] = phi float [ [[IND_END]], [[MIDDLE_BLOCK]] ], [ [[INIT]], [[FOR_BODY_LR_PH]] ]
; VEC1_INTERL2-NEXT:    br label [[FOR_BODY:%.*]]
; VEC1_INTERL2:       for.body:
; VEC1_INTERL2-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ], [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ]
; VEC1_INTERL2-NEXT:    [[X_05:%.*]] = phi float [ [[BC_RESUME_VAL1]], [[SCALAR_PH]] ], [ [[ADD:%.*]], [[FOR_BODY]] ]
; VEC1_INTERL2-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds float, float* [[A:%.*]], i64 [[INDVARS_IV]]
; VEC1_INTERL2-NEXT:    store float [[X_05]], float* [[ARRAYIDX]], align 4
; VEC1_INTERL2-NEXT:    [[ADD]] = fsub reassoc float [[X_05]], [[FPINC]]
; VEC1_INTERL2-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; VEC1_INTERL2-NEXT:    [[LFTR_WIDEIV:%.*]] = trunc i64 [[INDVARS_IV_NEXT]] to i32
; VEC1_INTERL2-NEXT:    [[EXITCOND:%.*]] = icmp eq i32 [[LFTR_WIDEIV]], [[N]]
; VEC1_INTERL2-NEXT:    br i1 [[EXITCOND]], label [[FOR_END_LOOPEXIT:%.*]], label [[FOR_BODY]]
; VEC1_INTERL2:       for.end.loopexit:
; VEC1_INTERL2-NEXT:    br label [[FOR_END]]
; VEC1_INTERL2:       for.end:
; VEC1_INTERL2-NEXT:    ret void
;
; VEC2_INTERL1_PRED_STORE-LABEL: @fp_iv_loop1_reassoc_FMF(
; VEC2_INTERL1_PRED_STORE-NEXT:  entry:
; VEC2_INTERL1_PRED_STORE-NEXT:    [[CMP4:%.*]] = icmp sgt i32 [[N:%.*]], 0
; VEC2_INTERL1_PRED_STORE-NEXT:    br i1 [[CMP4]], label [[FOR_BODY_LR_PH:%.*]], label [[FOR_END:%.*]]
; VEC2_INTERL1_PRED_STORE:       for.body.lr.ph:
; VEC2_INTERL1_PRED_STORE-NEXT:    [[FPINC:%.*]] = load float, float* @fp_inc, align 4
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP0:%.*]] = add i32 [[N]], -1
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP1:%.*]] = zext i32 [[TMP0]] to i64
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP2:%.*]] = add nuw nsw i64 [[TMP1]], 1
; VEC2_INTERL1_PRED_STORE-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp eq i32 [[TMP0]], 0
; VEC2_INTERL1_PRED_STORE-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[FOR_BODY:%.*]], label [[VECTOR_PH:%.*]]
; VEC2_INTERL1_PRED_STORE:       vector.ph:
; VEC2_INTERL1_PRED_STORE-NEXT:    [[N_VEC:%.*]] = and i64 [[TMP2]], 8589934590
; VEC2_INTERL1_PRED_STORE-NEXT:    [[CAST_CRD:%.*]] = sitofp i64 [[N_VEC]] to float
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP3:%.*]] = fmul reassoc float [[FPINC]], [[CAST_CRD]]
; VEC2_INTERL1_PRED_STORE-NEXT:    [[IND_END:%.*]] = fsub reassoc float [[INIT:%.*]], [[TMP3]]
; VEC2_INTERL1_PRED_STORE-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <2 x float> poison, float [[INIT]], i32 0
; VEC2_INTERL1_PRED_STORE-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <2 x float> [[DOTSPLATINSERT]], <2 x float> poison, <2 x i32> zeroinitializer
; VEC2_INTERL1_PRED_STORE-NEXT:    [[DOTSPLATINSERT2:%.*]] = insertelement <2 x float> poison, float [[FPINC]], i32 0
; VEC2_INTERL1_PRED_STORE-NEXT:    [[DOTSPLAT3:%.*]] = shufflevector <2 x float> [[DOTSPLATINSERT2]], <2 x float> poison, <2 x i32> zeroinitializer
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP4:%.*]] = fmul reassoc <2 x float> [[DOTSPLAT3]], <float 0.000000e+00, float 1.000000e+00>
; VEC2_INTERL1_PRED_STORE-NEXT:    [[INDUCTION:%.*]] = fsub reassoc <2 x float> [[DOTSPLAT]], [[TMP4]]
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP5:%.*]] = fmul reassoc float [[FPINC]], 2.000000e+00
; VEC2_INTERL1_PRED_STORE-NEXT:    [[DOTSPLATINSERT4:%.*]] = insertelement <2 x float> poison, float [[TMP5]], i32 0
; VEC2_INTERL1_PRED_STORE-NEXT:    [[DOTSPLAT5:%.*]] = shufflevector <2 x float> [[DOTSPLATINSERT4]], <2 x float> poison, <2 x i32> zeroinitializer
; VEC2_INTERL1_PRED_STORE-NEXT:    br label [[VECTOR_BODY:%.*]]
; VEC2_INTERL1_PRED_STORE:       vector.body:
; VEC2_INTERL1_PRED_STORE-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; VEC2_INTERL1_PRED_STORE-NEXT:    [[VEC_IND:%.*]] = phi <2 x float> [ [[INDUCTION]], [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[VECTOR_BODY]] ]
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP6:%.*]] = getelementptr inbounds float, float* [[A:%.*]], i64 [[INDEX]]
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP7:%.*]] = bitcast float* [[TMP6]] to <2 x float>*
; VEC2_INTERL1_PRED_STORE-NEXT:    store <2 x float> [[VEC_IND]], <2 x float>* [[TMP7]], align 4
; VEC2_INTERL1_PRED_STORE-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; VEC2_INTERL1_PRED_STORE-NEXT:    [[VEC_IND_NEXT]] = fsub reassoc <2 x float> [[VEC_IND]], [[DOTSPLAT5]]
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP8:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; VEC2_INTERL1_PRED_STORE-NEXT:    br i1 [[TMP8]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], [[LOOP4:!llvm.loop !.*]]
; VEC2_INTERL1_PRED_STORE:       middle.block:
; VEC2_INTERL1_PRED_STORE-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[TMP2]], [[N_VEC]]
; VEC2_INTERL1_PRED_STORE-NEXT:    br i1 [[CMP_N]], label [[FOR_END]], label [[FOR_BODY]]
; VEC2_INTERL1_PRED_STORE:       for.body:
; VEC2_INTERL1_PRED_STORE-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ], [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[FOR_BODY_LR_PH]] ]
; VEC2_INTERL1_PRED_STORE-NEXT:    [[X_05:%.*]] = phi float [ [[ADD:%.*]], [[FOR_BODY]] ], [ [[IND_END]], [[MIDDLE_BLOCK]] ], [ [[INIT]], [[FOR_BODY_LR_PH]] ]
; VEC2_INTERL1_PRED_STORE-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[INDVARS_IV]]
; VEC2_INTERL1_PRED_STORE-NEXT:    store float [[X_05]], float* [[ARRAYIDX]], align 4
; VEC2_INTERL1_PRED_STORE-NEXT:    [[ADD]] = fsub reassoc float [[X_05]], [[FPINC]]
; VEC2_INTERL1_PRED_STORE-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; VEC2_INTERL1_PRED_STORE-NEXT:    [[LFTR_WIDEIV:%.*]] = trunc i64 [[INDVARS_IV_NEXT]] to i32
; VEC2_INTERL1_PRED_STORE-NEXT:    [[EXITCOND:%.*]] = icmp eq i32 [[LFTR_WIDEIV]], [[N]]
; VEC2_INTERL1_PRED_STORE-NEXT:    br i1 [[EXITCOND]], label [[FOR_END]], label [[FOR_BODY]], [[LOOP5:!llvm.loop !.*]]
; VEC2_INTERL1_PRED_STORE:       for.end:
; VEC2_INTERL1_PRED_STORE-NEXT:    ret void
;
entry:
  %cmp4 = icmp sgt i32 %N, 0
  br i1 %cmp4, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %fpinc = load float, float* @fp_inc, align 4
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %x.05 = phi float [ %init, %for.body.lr.ph ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %A, i64 %indvars.iv
  store float %x.05, float* %arrayidx, align 4
  %add = fsub reassoc float %x.05, %fpinc
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

;void fp_iv_loop2(float init, float * __restrict__ A, int N) {
;  float x = init;
;  for (int i=0; i < N; ++i) {
;    A[i] = x;
;    x += 0.5;
;  }
;}

; VEC4_INTERL1-LABEL: @fp_iv_loop2(
; VEC4_INTERL1:       vector.ph:
; VEC4_INTERL1:         [[DOTSPLATINSERT:%.*]] = insertelement <4 x float> poison, float %init, i32 0
; VEC4_INTERL1-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <4 x float> [[DOTSPLATINSERT]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL1-NEXT:    [[INDUCTION2:%.*]] = fadd fast <4 x float> [[DOTSPLAT]], <float 0.000000e+00, float 5.000000e-01, float 1.000000e+00, float 1.500000e+00>
; VEC4_INTERL1-NEXT:    br label %vector.body
; VEC4_INTERL1:       vector.body:
; VEC4_INTERL1-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; VEC4_INTERL1-NEXT:    [[VEC_IND:%.*]] = phi <4 x float> [ [[INDUCTION2]], %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %vector.body ]
; VEC4_INTERL1-NEXT:    [[TMP7:%.*]] = getelementptr inbounds float, float* %A, i64 [[INDEX]]
; VEC4_INTERL1-NEXT:    [[TMP8:%.*]] = bitcast float* [[TMP7]] to <4 x float>*
; VEC4_INTERL1-NEXT:    store <4 x float> [[VEC_IND]], <4 x float>* [[TMP8]], align 4
; VEC4_INTERL1-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; VEC4_INTERL1-NEXT:    [[VEC_IND_NEXT]] = fadd fast <4 x float> [[VEC_IND]], <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00>
; VEC4_INTERL1:         br i1 {{.*}}, label %middle.block, label %vector.body

define void @fp_iv_loop2(float %init, float* noalias nocapture %A, i32 %N) #0 {
entry:
  %cmp4 = icmp sgt i32 %N, 0
  br i1 %cmp4, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %x.06 = phi float [ %conv1, %for.body ], [ %init, %for.body.preheader ]
  %arrayidx = getelementptr inbounds float, float* %A, i64 %indvars.iv
  store float %x.06, float* %arrayidx, align 4
  %conv1 = fadd fast float %x.06, 5.000000e-01
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

;void fp_iv_loop3(float init, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int N) {
;  int i = 0;
;  float x = init;
;  float y = 0.1;
;  for (; i < N; ++i) {
;    A[i] = x;
;    x += fp_inc;
;    y -= 0.5;
;    B[i] = x + y;
;    C[i] = y;
;  }
;}

; VEC4_INTERL1-LABEL: @fp_iv_loop3(
; VEC4_INTERL1:       for.body.lr.ph:
; VEC4_INTERL1:         [[TMP0:%.*]] = load float, float* @fp_inc, align 4
; VEC4_INTERL1:       vector.ph:
; VEC4_INTERL1:         [[DOTSPLATINSERT:%.*]] = insertelement <4 x float> poison, float %init, i32 0
; VEC4_INTERL1-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <4 x float> [[DOTSPLATINSERT]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL1-NEXT:    [[DOTSPLATINSERT5:%.*]] = insertelement <4 x float> poison, float [[TMP0]], i32 0
; VEC4_INTERL1-NEXT:    [[DOTSPLAT6:%.*]] = shufflevector <4 x float> [[DOTSPLATINSERT5]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL1-NEXT:    [[TMP7:%.*]] = fmul fast <4 x float> [[DOTSPLAT6]], <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>
; VEC4_INTERL1-NEXT:    [[INDUCTION7:%.*]] = fadd fast <4 x float> [[DOTSPLAT]], [[TMP7]]
; VEC4_INTERL1-NEXT:    [[TMP8:%.*]] = fmul fast float [[TMP0]], 4.000000e+00
; VEC4_INTERL1-NEXT:    [[DOTSPLATINSERT8:%.*]] = insertelement <4 x float> poison, float [[TMP8]], i32 0
; VEC4_INTERL1-NEXT:    [[DOTSPLAT9:%.*]] = shufflevector <4 x float> [[DOTSPLATINSERT8]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL1-NEXT:    [[BROADCAST_SPLATINSERT12:%.*]] = insertelement <4 x float> poison, float [[TMP0]], i32 0
; VEC4_INTERL1-NEXT:    [[BROADCAST_SPLAT13:%.*]] = shufflevector <4 x float> [[BROADCAST_SPLATINSERT12]], <4 x float> poison, <4 x i32> zeroinitializer
; VEC4_INTERL1-NEXT:    br label [[VECTOR_BODY:%.*]]
; VEC4_INTERL1:       vector.body:
; VEC4_INTERL1-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; VEC4_INTERL1-NEXT:    [[VEC_IND:%.*]] = phi <4 x float> [ <float 0x3FB99999A0000000, float 0xBFD99999A0000000, float 0xBFECCCCCC0000000, float 0xBFF6666660000000>, %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %vector.body ]
; VEC4_INTERL1-NEXT:    [[VEC_IND10:%.*]] = phi <4 x float> [ [[INDUCTION7]], %vector.ph ], [ [[VEC_IND_NEXT11:%.*]], %vector.body ]
; VEC4_INTERL1-NEXT:    [[TMP12:%.*]] = getelementptr inbounds float, float* %A, i64 [[INDEX]]
; VEC4_INTERL1-NEXT:    [[TMP13:%.*]] = bitcast float* [[TMP12]] to <4 x float>*
; VEC4_INTERL1-NEXT:    store <4 x float> [[VEC_IND10]], <4 x float>* [[TMP13]], align 4
; VEC4_INTERL1-NEXT:    [[TMP14:%.*]] = fadd fast <4 x float> [[VEC_IND10]], [[BROADCAST_SPLAT13]]
; VEC4_INTERL1-NEXT:    [[TMP15:%.*]] = fadd fast <4 x float> [[VEC_IND]], <float -5.000000e-01, float -5.000000e-01, float -5.000000e-01, float -5.000000e-01>
; VEC4_INTERL1-NEXT:    [[TMP16:%.*]] = fadd fast <4 x float> [[TMP15]], [[TMP14]]
; VEC4_INTERL1-NEXT:    [[TMP17:%.*]] = getelementptr inbounds float, float* %B, i64 [[INDEX]]
; VEC4_INTERL1-NEXT:    [[TMP18:%.*]] = bitcast float* [[TMP17]] to <4 x float>*
; VEC4_INTERL1-NEXT:    store <4 x float> [[TMP16]], <4 x float>* [[TMP18]], align 4
; VEC4_INTERL1-NEXT:    [[TMP19:%.*]] = getelementptr inbounds float, float* %C, i64 [[INDEX]]
; VEC4_INTERL1-NEXT:    [[TMP20:%.*]] = bitcast float* [[TMP19]] to <4 x float>*
; VEC4_INTERL1-NEXT:    store <4 x float> [[TMP15]], <4 x float>* [[TMP20]], align 4
; VEC4_INTERL1-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; VEC4_INTERL1-NEXT:    [[VEC_IND_NEXT]] = fadd fast <4 x float> [[VEC_IND]], <float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00>
; VEC4_INTERL1-NEXT:    [[VEC_IND_NEXT11]] = fadd fast <4 x float> [[VEC_IND10]], [[DOTSPLAT9]]
; VEC4_INTERL1:         br i1 {{.*}}, label %middle.block, label %vector.body

define void @fp_iv_loop3(float %init, float* noalias nocapture %A, float* noalias nocapture %B, float* noalias nocapture %C, i32 %N) {
entry:
  %cmp9 = icmp sgt i32 %N, 0
  br i1 %cmp9, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %0 = load float, float* @fp_inc, align 4
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %y.012 = phi float [ 0x3FB99999A0000000, %for.body.lr.ph ], [ %conv1, %for.body ]
  %x.011 = phi float [ %init, %for.body.lr.ph ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %A, i64 %indvars.iv
  store float %x.011, float* %arrayidx, align 4
  %add = fadd fast float %x.011, %0
  %conv1 = fadd fast float %y.012, -5.000000e-01
  %add2 = fadd fast float %conv1, %add
  %arrayidx4 = getelementptr inbounds float, float* %B, i64 %indvars.iv
  store float %add2, float* %arrayidx4, align 4
  %arrayidx6 = getelementptr inbounds float, float* %C, i64 %indvars.iv
  store float %conv1, float* %arrayidx6, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

; Start and step values are constants. There is no 'fmul' operation in this case
;void fp_iv_loop4(float * __restrict__ A, int N) {
;  float x = 1.0;
;  for (int i=0; i < N; ++i) {
;    A[i] = x;
;    x += 0.5;
;  }
;}

; VEC4_INTERL1-LABEL: @fp_iv_loop4(
; VEC4_INTERL1:       vector.ph:
; VEC4_INTERL1:         br label %vector.body
; VEC4_INTERL1:       vector.body:
; VEC4_INTERL1-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; VEC4_INTERL1-NEXT:    [[VEC_IND:%.*]] = phi <4 x float> [ <float 1.000000e+00, float 1.500000e+00, float 2.000000e+00, float 2.500000e+00>, %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %vector.body ]
; VEC4_INTERL1-NEXT:    [[TMP7:%.*]] = getelementptr inbounds float, float* %A, i64 [[INDEX]]
; VEC4_INTERL1-NEXT:    [[TMP8:%.*]] = bitcast float* [[TMP7]] to <4 x float>*
; VEC4_INTERL1-NEXT:    store <4 x float> [[VEC_IND]], <4 x float>* [[TMP8]], align 4
; VEC4_INTERL1-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; VEC4_INTERL1-NEXT:    [[VEC_IND_NEXT]] = fadd fast <4 x float> [[VEC_IND]], <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00>
; VEC4_INTERL1:         br i1 {{.*}}, label %middle.block, label %vector.body

define void @fp_iv_loop4(float* noalias nocapture %A, i32 %N) {
entry:
  %cmp4 = icmp sgt i32 %N, 0
  br i1 %cmp4, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %x.06 = phi float [ %conv1, %for.body ], [ 1.000000e+00, %for.body.preheader ]
  %arrayidx = getelementptr inbounds float, float* %A, i64 %indvars.iv
  store float %x.06, float* %arrayidx, align 4
  %conv1 = fadd fast float %x.06, 5.000000e-01
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

; VEC2_INTERL1_PRED_STORE-LABEL: @non_primary_iv_float_scalar(
; VEC2_INTERL1_PRED_STORE:       vector.body:
; VEC2_INTERL1_PRED_STORE-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %[[PRED_STORE_CONTINUE7:.*]] ]
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP1:%.*]] = sitofp i64 [[INDEX]] to float
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP2:%.*]] = getelementptr inbounds float, float* %A, i64 [[INDEX]]
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP3:%.*]] = bitcast float* [[TMP2]] to <2 x float>*
; VEC2_INTERL1_PRED_STORE-NEXT:    [[WIDE_LOAD:%.*]] = load <2 x float>, <2 x float>* [[TMP3]], align 4
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP4:%.*]] = fcmp fast oeq <2 x float> [[WIDE_LOAD]], zeroinitializer
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP5:%.*]] = extractelement <2 x i1> [[TMP4]], i32 0
; VEC2_INTERL1_PRED_STORE-NEXT:    br i1 [[TMP5]], label %[[PRED_STORE_IF:.*]], label %[[PRED_STORE_CONTINUE:.*]]
; VEC2_INTERL1_PRED_STORE:       [[PRED_STORE_IF]]:
; VEC2_INTERL1_PRED_STORE-NEXT:    store float [[TMP1]], float* [[TMP2]], align 4
; VEC2_INTERL1_PRED_STORE-NEXT:    br label %[[PRED_STORE_CONTINUE]]
; VEC2_INTERL1_PRED_STORE:       [[PRED_STORE_CONTINUE]]:
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP8:%.*]] = extractelement <2 x i1> [[TMP4]], i32 1
; VEC2_INTERL1_PRED_STORE-NEXT:    br i1 [[TMP8]], label %[[PRED_STORE_IF6:.*]], label %[[PRED_STORE_CONTINUE7]]
; VEC2_INTERL1_PRED_STORE:       [[PRED_STORE_IF6]]:
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP9:%.*]] = fadd fast float [[TMP1]], 1.000000e+00
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP10:%.*]] = or i64 [[INDEX]], 1
; VEC2_INTERL1_PRED_STORE-NEXT:    [[TMP11:%.*]] = getelementptr inbounds float, float* %A, i64 [[TMP10]]
; VEC2_INTERL1_PRED_STORE-NEXT:    store float [[TMP9]], float* [[TMP11]], align 4
; VEC2_INTERL1_PRED_STORE-NEXT:    br label %[[PRED_STORE_CONTINUE7]]
; VEC2_INTERL1_PRED_STORE:       [[PRED_STORE_CONTINUE7]]:
; VEC2_INTERL1_PRED_STORE-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; VEC2_INTERL1_PRED_STORE:         br i1 {{.*}}, label %middle.block, label %vector.body

define void @non_primary_iv_float_scalar(float* %A, i64 %N) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.inc ], [ 0, %entry ]
  %j = phi float [ %j.next, %for.inc ], [ 0.0, %entry ]
  %tmp0 = getelementptr inbounds float, float* %A, i64 %i
  %tmp1 = load float, float* %tmp0, align 4
  %tmp2 = fcmp fast oeq float %tmp1, 0.0
  br i1 %tmp2, label %if.pred, label %for.inc

if.pred:
  store float %j, float* %tmp0, align 4
  br label %for.inc

for.inc:
  %i.next = add nuw nsw i64 %i, 1
  %j.next = fadd fast float %j, 1.0
  %cond = icmp slt i64 %i.next, %N
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

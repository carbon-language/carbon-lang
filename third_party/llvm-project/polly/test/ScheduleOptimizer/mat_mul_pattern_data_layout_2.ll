; RUN: opt %loadPolly -polly-pattern-matching-based-opts=true \
; RUN: -polly-target-throughput-vector-fma=1 \
; RUN: -polly-target-latency-vector-fma=8 \
; RUN: -polly-target-1st-cache-level-associativity=8 \
; RUN: -polly-target-2nd-cache-level-associativity=8 \
; RUN: -polly-target-1st-cache-level-size=32768 \
; RUN: -polly-target-2nd-cache-level-size=262144 \
; RUN: -polly-target-vector-register-bitwidth=256 \
; RUN: -polly-opt-isl -polly-print-ast -disable-output < %s | FileCheck %s
;
;    /* C := alpha*A*B + beta*C */
;    /* _PB_NK % Kc != 0 */
;    for (i = 0; i < _PB_NI; i++)
;      for (j = 0; j < _PB_NJ; j++)
;        {
;	   C[i][j] *= beta;
;	   for (k = 0; k < _PB_NK; ++k)
;	     C[i][j] += alpha * A[i][k] * B[k][j];
;        }
;
; CHECK-LABEL: Printing analysis 'Polly - Generate an AST from the SCoP (isl)' for region: 'bb8 => bb32' in function 'kernel_gemm':
; CHECK:    {
; CHECK-NEXT:      // 1st level tiling - Tiles
; CHECK-NEXT:      for (int c0 = 0; c0 <= 32; c0 += 1)
; CHECK-NEXT:        for (int c1 = 0; c1 <= 32; c1 += 1) {
; CHECK-NEXT:          // 1st level tiling - Points
; CHECK-NEXT:          for (int c2 = 0; c2 <= 31; c2 += 1)
; CHECK-NEXT:            for (int c3 = 0; c3 <= 31; c3 += 1)
; CHECK-NEXT:              Stmt_bb9(32 * c0 + c2, 32 * c1 + c3);
; CHECK-NEXT:        }
; CHECK-NEXT:      // 1st level tiling - Tiles
; CHECK-NEXT:      for (int c1 = 0; c1 <= 3; c1 += 1) {
; CHECK-NEXT:        for (int c3 = 0; c3 <= 1055; c3 += 1)
; CHECK-NEXT:          for (int c4 = 256 * c1; c4 <= min(1022, 256 * c1 + 255); c4 += 1)
; CHECK-NEXT:            CopyStmt_0(0, c3, c4);
; CHECK-NEXT:        for (int c2 = 0; c2 <= 10; c2 += 1) {
; CHECK-NEXT:          for (int c6 = 96 * c2; c6 <= 96 * c2 + 95; c6 += 1)
; CHECK-NEXT:            for (int c7 = 256 * c1; c7 <= min(1022, 256 * c1 + 255); c7 += 1)
; CHECK-NEXT:              CopyStmt_1(0, c1, c2, c6, c7);
; CHECK-NEXT:          // 1st level tiling - Points
; CHECK-NEXT:          // Register tiling - Tiles
; CHECK-NEXT:          for (int c3 = 0; c3 <= 131; c3 += 1)
; CHECK-NEXT:            for (int c4 = 0; c4 <= 23; c4 += 1)
; CHECK-NEXT:              for (int c5 = 0; c5 <= min(255, -256 * c1 + 1022); c5 += 1) {
; CHECK-NEXT:                // Loop Vectorizer Disabled
; CHECK-NEXT:                // Register tiling - Points
; CHECK-NEXT:                {
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4, 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4, 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4, 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4, 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4, 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4, 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4, 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4, 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 1, 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 1, 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 1, 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 1, 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 1, 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 1, 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 1, 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 1, 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 2, 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 2, 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 2, 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 2, 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 2, 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 2, 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 2, 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 2, 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 3, 8 * c3, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 3, 8 * c3 + 1, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 3, 8 * c3 + 2, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 3, 8 * c3 + 3, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 3, 8 * c3 + 4, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 3, 8 * c3 + 5, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 3, 8 * c3 + 6, 256 * c1 + c5);
; CHECK-NEXT:                  Stmt_Copy_0(96 * c2 + 4 * c4 + 3, 8 * c3 + 7, 256 * c1 + c5);
; CHECK-NEXT:                }
; CHECK-NEXT:              }
; CHECK-NEXT:        }
; CHECK-NEXT:      }
; CHECK-NEXT:    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define internal void @kernel_gemm(i32 %arg, i32 %arg1, i32 %arg2, double %arg3, double %arg4, [1056 x double]* %arg5, [1023 x double]* %arg6, [1056 x double]* %arg7) #0 {
bb:
  br label %bb8

bb8:                                              ; preds = %bb29, %bb
  %tmp = phi i64 [ 0, %bb ], [ %tmp30, %bb29 ]
  br label %bb9

bb9:                                              ; preds = %bb26, %bb8
  %tmp10 = phi i64 [ 0, %bb8 ], [ %tmp27, %bb26 ]
  %tmp11 = getelementptr inbounds [1056 x double], [1056 x double]* %arg5, i64 %tmp, i64 %tmp10
  %tmp12 = load double, double* %tmp11, align 8
  %tmp13 = fmul double %tmp12, %arg4
  store double %tmp13, double* %tmp11, align 8
  br label %Copy_0

Copy_0:                                             ; preds = %Copy_0, %bb9
  %tmp15 = phi i64 [ 0, %bb9 ], [ %tmp24, %Copy_0 ]
  %tmp16 = getelementptr inbounds [1023 x double], [1023 x double]* %arg6, i64 %tmp, i64 %tmp15
  %tmp17 = load double, double* %tmp16, align 8
  %tmp18 = fmul double %tmp17, %arg3
  %tmp19 = getelementptr inbounds [1056 x double], [1056 x double]* %arg7, i64 %tmp15, i64 %tmp10
  %tmp20 = load double, double* %tmp19, align 8
  %tmp21 = fmul double %tmp18, %tmp20
  %tmp22 = load double, double* %tmp11, align 8
  %tmp23 = fadd double %tmp22, %tmp21
  store double %tmp23, double* %tmp11, align 8
  %tmp24 = add nuw nsw i64 %tmp15, 1
  %tmp25 = icmp ne i64 %tmp24, 1023
  br i1 %tmp25, label %Copy_0, label %bb26

bb26:                                             ; preds = %Copy_0
  %tmp27 = add nuw nsw i64 %tmp10, 1
  %tmp28 = icmp ne i64 %tmp27, 1056
  br i1 %tmp28, label %bb9, label %bb29

bb29:                                             ; preds = %bb26
  %tmp30 = add nuw nsw i64 %tmp, 1
  %tmp31 = icmp ne i64 %tmp30, 1056
  br i1 %tmp31, label %bb8, label %bb32

bb32:                                             ; preds = %bb29
  ret void
}

attributes #0 = { nounwind uwtable "target-cpu"="x86-64" "target-features"="+aes,+avx,+cmov,+cx16,+fxsr,+mmx,+pclmul,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt" }

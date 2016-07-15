; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; REQUIRES: pollyacc
;
; CODE: Code
; CODE-NEXT: ====
; CODE-NEXT: # host
; CODE-NEXT: {
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (1024) * sizeof(float), cudaMemcpyHostToDevice));
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_b, &MemRef_b, sizeof(float), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(32);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> ();
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (1024) * sizeof(float), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: Stmt_bb2(32 * b0 + t0);

;    void foo(float A[], float b) {
;      for (long i = 0; i < 1024; i++)
;        A[i] += b;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(float* %A, float %b) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb5, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp6, %bb5 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb7

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp3 = load float, float* %tmp, align 4
  %tmp4 = fadd float %tmp3, %b
  store float %tmp4, float* %tmp, align 4
  br label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb7:                                              ; preds = %bb1
  ret void
}

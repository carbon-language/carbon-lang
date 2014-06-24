; REQUIRES: nvptx-registered-target
; RUN: opt %loadPolly -basicaa -polly-import-jscop -polly-import-jscop-dir=%S -polly-import-jscop-postfix=transformed+gpu -enable-polly-gpgpu -polly-gpgpu-triple=nvptx64-unknown-unknown -polly-codegen < %s -S | FileCheck %s

;int A[1024];

;int gpu() {
;  int i;
;
;  for(i = 0; i < 1024; i++)
;    A[i] = i*128 + 508;
;
;  return 0;
;}
;
;int main() {
;  int b = gpu();
;  return 0;
;}

; ModuleID = '1d_parallel.s'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [1024 x i32] zeroinitializer, align 16

; Function Attrs: nounwind uwtable
define i32 @gpu() #0 {
  br label %.split

.split:                                           ; preds = %0
  br label %1

; <label>:1                                       ; preds = %.split, %1
  %indvar = phi i64 [ 0, %.split ], [ %indvar.next, %1 ]
  %2 = mul i64 %indvar, 128
  %3 = add i64 %2, 508
  %4 = trunc i64 %3 to i32
  %scevgep = getelementptr [1024 x i32]* @A, i64 0, i64 %indvar
  store i32 %4, i32* %scevgep, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar.next, 1024
  br i1 %exitcond, label %1, label %5

; <label>:5                                       ; preds = %1
  ret i32 0
}

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
  br label %.split

.split:                                           ; preds = %0
  %1 = tail call i32 @gpu()
  ret i32 0
}

; CHECK:  call void @polly_initDevice
; CHECK:  call void @polly_getPTXModule
; CHECK:  call void @polly_getPTXKernelEntry
; CHECK:  call void @polly_allocateMemoryForHostAndDevice
; CHECK:  call void @polly_setKernelParameters
; CHECK:  call void @polly_startTimerByCudaEvent
; CHECK:  call void @polly_launchKernel
; CHECK:  call void @polly_copyFromDeviceToHost
; CHECK:  call void @polly_stopTimerByCudaEvent
; CHECK:  call void @polly_cleanupGPGPUResources

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = metadata !{metadata !"clang version 3.5.0 "}

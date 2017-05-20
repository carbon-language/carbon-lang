; RUN: opt < %s -S  -inferattrs -licm | FileCheck %s

; CHECK-LABEL: define void @pthread_self_safe(
; CHECK-NEXT: call i64 @pthread_self() 
define void @pthread_self_safe(i32) {
  br label %2

; <label>:2:                                      ; preds = %7, %1
  %idx = phi i32 [ 0, %1 ], [ %8, %7 ]
  %3 = icmp slt i32 %idx, %0
  br i1 %3, label %4, label %9

; <label>:4:                                      ; preds = %2
  call void @external_func_that_could_do_anything()
  %5 = call i64 @pthread_self() #1
  %6 = trunc i64 %5 to i32
  call void @use_pthread_self(i32 %6)
  br label %7

; <label>:7:                                      ; preds = %4
  %8 = add nsw i32 %idx, 1
  br label %2

; <label>:9:                                      ; preds = %2
  ret void
}

; CHECK: declare i64 @pthread_self() #0
; CHECK: attributes #0 = { nounwind readnone speculatable }
; Function Attrs: nounwind readnone
declare i64 @pthread_self() #1

declare void @external_func_that_could_do_anything()

declare void @use_pthread_self(i32)

attributes #1 = { nounwind readnone }


; RUN: opt -mtriple=nvptx64-nvidia-cuda -load-store-vectorizer -S -o - %s | FileCheck %s

; If we have a chain of loads or stores with a side-effecting operation in the
; middle, we should still be able to merge the loads/stores that appear
; before/after the side-effecting op.  We just can't merge *across* the
; side-effecting op.

declare void @fn() #0

; CHECK-LABEL: @merge_stores
; CHECK: store <2 x i32> <i32 100, i32 101>
; CHECK: call void @fn()
; CHECK: store <2 x i32> <i32 102, i32 103>
define void @merge_stores(i32* %out) #0 {
  %out.gep.1 = getelementptr i32, i32* %out, i32 1
  %out.gep.2 = getelementptr i32, i32* %out, i32 2
  %out.gep.3 = getelementptr i32, i32* %out, i32 3

  store i32 101, i32* %out.gep.1
  store i32 100, i32* %out
  call void @fn()
  store i32 102, i32* %out.gep.2
  store i32 103, i32* %out.gep.3
  ret void
}

; CHECK-LABEL: @merge_loads
; CHECK: load <2 x i32>
; CHECK: call void @fn()
; CHECK: load <2 x i32>
define i32 @merge_loads(i32* %in) #0 {
  %in.gep.1 = getelementptr i32, i32* %in, i32 1
  %in.gep.2 = getelementptr i32, i32* %in, i32 2
  %in.gep.3 = getelementptr i32, i32* %in, i32 3

  %v1 = load i32, i32* %in
  %v2 = load i32, i32* %in.gep.1
  call void @fn()
  %v3 = load i32, i32* %in.gep.2
  %v4 = load i32, i32* %in.gep.3

  %sum1 = add i32 %v1, %v2
  %sum2 = add i32 %sum1, %v3
  %sum3 = add i32 %sum2, %v4
  ret i32 %v4
}

attributes #0 = { nounwind }

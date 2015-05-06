; RUN: llc < %s | FileCheck %s

declare i1024 @g()

define i1024 @f() gc "statepoint-example" {
; CHECK-LABEL: _f
; CHECK: callq _g
  %1 = invoke i32 (i1024 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1024f(i1024 ()* @g, i32 0, i32 0, i32 0)
          to label %normal unwind label %except

normal:                                           ; preds = %0
  %x1 = call i1024 @llvm.experimental.gc.result.i1024(i32 %1)
  ret i1024 %x1

except:                                           ; preds = %0
  %landing_pad = landingpad { i8*, i32 } personality i32 ()* @personality_function
          cleanup
  ret i1024 0
}

declare i32 @personality_function()

; Function Attrs: nounwind
declare i32 @llvm.experimental.gc.statepoint.p0f_i1024f(i1024 ()*, i32, i32, ...) #0

; Function Attrs: nounwind
declare i1024 @llvm.experimental.gc.result.i1024(i32) #0

attributes #0 = { nounwind }

; LLI.exe used to crash on Windows\X86 when certain single precession
; floating point intrinsics (defined as macros) are used.
; This unit test guards against the failure.
;
; RUN: %lli %s | FileCheck %s

@flt = internal global float 12.0e+0
@str = internal constant [18 x i8] c"Double value: %f\0A\00"

declare i32 @printf(i8* nocapture, ...) nounwind

define i32 @main() {
  %flt = load float* @flt
  %float2 = frem float %flt, 5.0
  %double1 = fpext float %float2 to double
  call i32 (i8*, ...)* @printf(i8* getelementptr ([18 x i8]* @str, i32 0, i64 0), double %double1)
  ret i32 0
}

; CHECK: Double value: 2.0

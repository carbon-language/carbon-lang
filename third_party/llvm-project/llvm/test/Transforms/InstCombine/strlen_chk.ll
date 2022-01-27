; Test that __strlen_chk simplification works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [6 x i8] c"hello\00"
@hello_no_nul = constant [5 x i8] c"hello"

declare i32 @__strlen_chk(i8*, i32)

; Check __strlen_chk(string constant) -> strlen or constants

; CHECK-LABEL: @unknown_str_known_object_size
define i32 @unknown_str_known_object_size(i8* %c) {
  ; CHECK: call i32 @__strlen_chk
  %1 = call i32 @__strlen_chk(i8* %c, i32 8)
  ret i32 %1
}

; CHECK-LABEL: @known_str_known_object_size
define i32 @known_str_known_object_size(i8* %c) {
  ; CHECK: ret i32 5
  %1 = call i32 @__strlen_chk(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @hello, i32 0, i32 0), i32 6)
  ret i32 %1
}

; CHECK-LABEL: @known_str_too_small_object_size
define i32 @known_str_too_small_object_size(i8* %c) {
  ; CHECK: call i32 @__strlen_chk
  %1 = call i32 @__strlen_chk(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @hello, i32 0, i32 0), i32 5)
  ret i32 %1
}

; CHECK-LABEL: @known_str_no_nul
define i32 @known_str_no_nul(i8* %c) {
  ; CHECK: call i32 @__strlen_chk
  %1 = call i32 @__strlen_chk(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @hello_no_nul, i32 0, i32 0), i32 5)
  ret i32 %1
}

; CHECK-LABEL: @unknown_str_unknown_object_size
define i32 @unknown_str_unknown_object_size(i8* %c) {
  ; CHECK: call i32 @strlen
  %1 = call i32 @__strlen_chk(i8* %c, i32 -1)
  ret i32 %1
}

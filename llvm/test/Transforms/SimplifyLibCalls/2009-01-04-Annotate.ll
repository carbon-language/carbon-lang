; RUN: opt < %s -simplify-libcalls -S | FileCheck %s

; CHECK: declare noalias i8* @fopen(i8* nocapture, i8* nocapture) nounwind
declare i8* @fopen(i8*, i8*)

; CHECK: declare i8 @strlen(i8* nocapture) nounwind readonly
declare i8 @strlen(i8*)

; CHECK: declare noalias i32* @realloc(i32* nocapture, i32) nounwind
declare i32* @realloc(i32*, i32)

; Test deliberately wrong declaration
declare i32 @strcpy(...)

; CHECK-NOT: strcpy{{.*}}noalias
; CHECK-NOT: strcpy{{.*}}nocapture
; CHECK-NOT: strcpy{{.*}}nounwind
; CHECK-NOT: strcpy{{.*}}readonly

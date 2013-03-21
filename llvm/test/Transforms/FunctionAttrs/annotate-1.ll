; RUN: opt < %s -functionattrs -S | FileCheck %s

declare i8* @fopen(i8*, i8*)
; CHECK: declare noalias i8* @fopen(i8* nocapture, i8* nocapture) [[G0:#[0-9]]] 

declare i8 @strlen(i8*)
; CHECK: declare i8 @strlen(i8* nocapture) [[G1:#[0-9]]]

declare i32* @realloc(i32*, i32)
; CHECK: declare noalias i32* @realloc(i32* nocapture, i32) [[G0]]

; Test deliberately wrong declaration

declare i32 @strcpy(...)
; CHECK: declare i32 @strcpy(...)

; CHECK: attributes [[G0]] = { nounwind }
; CHECK: attributes [[G1]] = { nounwind readonly }

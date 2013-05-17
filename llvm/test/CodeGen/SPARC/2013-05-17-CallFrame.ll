; RUN: llc -march=sparc < %s | FileCheck %s

; CHECK: variable_alloca_with_adj_call_stack
; CHECK: save %sp, -96, %sp
; CHECK: add %sp, -16, %sp
; CHECK: call foo
; CHECK: add %sp, 16, %sp
define void @variable_alloca_with_adj_call_stack(i32 %num) {
entry:
  %0 = alloca i8, i32 %num, align 8
  call void @foo(i8* %0, i8* %0, i8* %0, i8* %0, i8* %0, i8* %0, i8* %0, i8* %0, i8* %0, i8* %0)
  ret void
}


declare void @foo(i8* , i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*);

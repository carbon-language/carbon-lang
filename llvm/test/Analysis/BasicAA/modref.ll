; RUN: opt < %s -basicaa -gvn -S | FileCheck %s

declare void @llvm.memset.i32(i8*, i8, i32, i32)

declare void @external(i32*) 

define i32 @test0(i8* %P) {
  %A = alloca i32
  call void @external(i32* %A)
  
  store i32 0, i32* %A
  
  call void @llvm.memset.i32(i8* %P, i8 0, i32 42, i32 1)
  
  %B = load i32* %A
  ret i32 %B
  
; CHECK: @test0
; CHECK: ret i32 0
}


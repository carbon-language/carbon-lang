; RUN: opt -S -gvn < %s | FileCheck %s

declare void @llvm.memset.i8(i8*, i8, i8, i32)

define i8 @test(i8* %P) {
  %P2 = getelementptr i8* %P, i32 1000
  store i8 1, i8* %P2  ;; Not dead across memset
  call void @llvm.memset.i8(i8* %P, i8 2, i8 127, i32 0)
  %A = load i8* %P2
  ret i8 %A
; CHECK: ret i8 1
}


; RUN: opt -gvn -instcombine -S < %s | FileCheck %s

declare i8 @llvm.atomic.load.add.i8.p0i8(i8*, i8)

define i8 @foo(i8* %ptr) {
  %P = getelementptr i8* %ptr, i32 0
  %Q = getelementptr i8* %ptr, i32 1
; CHECK: getelementptr
  %X = load i8* %P
  %Y = call i8 @llvm.atomic.load.add.i8.p0i8(i8* %Q, i8 1)
  %Z = load i8* %P
; CHECK-NOT: = load
  %A = sub i8 %X, %Z
  ret i8 %A
; CHECK: ret i8 0
}

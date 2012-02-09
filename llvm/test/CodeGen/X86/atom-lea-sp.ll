; RUN: llc < %s -mcpu=atom -mtriple=i686-linux  | FileCheck -check-prefix=atom %s
; RUN: llc < %s -mcpu=core2 -mtriple=i686-linux | FileCheck %s

declare void @use_arr(i8*)
declare void @many_params(i32, i32, i32, i32, i32, i32)

define void @test1() nounwind {
; atom: test1:
; atom: leal -1052(%esp), %esp
; atom-NOT: sub
; atom: call
; atom: leal 1052(%esp), %esp

; CHECK: test1:
; CHECK: subl
; CHECK: call
; CHECK-NOT: lea
  %arr = alloca [1024 x i8], align 16
  %arr_ptr = getelementptr inbounds [1024 x i8]* %arr, i8 0, i8 0
  call void @use_arr(i8* %arr_ptr)
  ret void
}

define void @test2() nounwind {
; atom: test2:
; atom: leal -28(%esp), %esp
; atom: call
; atom: leal 28(%esp), %esp

; CHECK: test2:
; CHECK-NOT: lea
  call void @many_params(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6)
  ret void
}

define void @test3() nounwind {
; atom: test3:
; atom: leal -8(%esp), %esp
; atom: leal 8(%esp), %esp

; CHECK: test3:
; CHECK-NOT: lea
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 0, i32* %x, align 4
  ret void
}


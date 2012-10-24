; RUN: opt -instcombine %s | llvm-dis | FileCheck %s
target datalayout = "e-p:32:32:32-p1:64:64:64-p2:8:8:8-p3:16:16:16--p4:96:96:96-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32"

define i32 @test_as0(i32 addrspace(0)* %A) {
entry:
; CHECK: %arrayidx = getelementptr i32* %A, i32 1
  %arrayidx = getelementptr i32 addrspace(0)* %A, i64 1
  %y = load i32 addrspace(0)* %arrayidx, align 4
  ret i32 %y
}

define i32 @test_as1(i32 addrspace(1)* %A) {
entry:
; CHECK: %arrayidx = getelementptr i32 addrspace(1)* %A, i64 1
  %arrayidx = getelementptr i32 addrspace(1)* %A, i32 1
  %y = load i32 addrspace(1)* %arrayidx, align 4
  ret i32 %y
}

define i32 @test_as2(i32 addrspace(2)* %A) {
entry:
; CHECK: %arrayidx = getelementptr i32 addrspace(2)* %A, i8 1
  %arrayidx = getelementptr i32 addrspace(2)* %A, i32 1
  %y = load i32 addrspace(2)* %arrayidx, align 4
  ret i32 %y
}

define i32 @test_as3(i32 addrspace(3)* %A) {
entry:
; CHECK: %arrayidx = getelementptr i32 addrspace(3)* %A, i16 1
  %arrayidx = getelementptr i32 addrspace(3)* %A, i32 1
  %y = load i32 addrspace(3)* %arrayidx, align 4
  ret i32 %y
}

define i32 @test_as4(i32 addrspace(4)* %A) {
entry:
; CHECK: %arrayidx = getelementptr i32 addrspace(4)* %A, i96 1
  %arrayidx = getelementptr i32 addrspace(4)* %A, i32 1
  %y = load i32 addrspace(4)* %arrayidx, align 4
  ret i32 %y
}


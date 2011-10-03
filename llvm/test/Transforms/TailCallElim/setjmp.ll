; RUN: opt < %s -tailcallelim -S | FileCheck %s

; Test that we don't tail call in a functions that calls setjmp.

; CHECK-NOT: tail call void @bar()

define void @foo(i32* %x) {
bb:
  %tmp75 = tail call i32 @setjmp(i32* %x)
  call void @bar()
  ret void
}

declare i32 @setjmp(i32*) returns_twice

declare void @bar()

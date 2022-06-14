; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: insertvalue operand and field disagree in type: 'i8*' instead of 'i32'

define void @test() {
entry:
  insertvalue { i32, i32 } undef, i8* null, 0
  ret void
}

; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: insertvalue operand and field disagree in type: 'i32' instead of 'i64'

define <{ i32 }> @test() {
  ret <{ i32 }> insertvalue (<{ i64 }> zeroinitializer, i32 4, 0)
}

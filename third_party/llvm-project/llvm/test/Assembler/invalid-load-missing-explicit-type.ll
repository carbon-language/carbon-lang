; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; CHECK: <stdin>:4:18: error: expected comma after load's type
define void @test(i32* %t) {
  %x = load i32* %t
  ret void
}

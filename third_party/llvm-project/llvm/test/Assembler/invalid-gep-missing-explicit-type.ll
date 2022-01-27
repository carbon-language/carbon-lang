; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; CHECK: <stdin>:4:27: error: expected comma after getelementptr's type
define void @test(i32* %t) {
  %x = getelementptr i32* %t, i32 0
  ret void
}


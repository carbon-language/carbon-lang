; RUN: not llvm-as < %s |& FileCheck %s

; CHECK: invalid indices for insertvalue

define void @test() {
entry:
  insertvalue [0 x i32] undef, i32 0, 0
  ret void
}

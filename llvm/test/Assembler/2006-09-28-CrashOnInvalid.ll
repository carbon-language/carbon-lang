; Test for PR902.  This program is erroneous, but should not crash llvm-as.
; This tests that a simple error is caught and processed correctly.
; RUN: llvm-as < %s >/dev/null |& grep {FP constant invalid for type}

define void @test() {
  add i32 1, 2.0
  ret void
}

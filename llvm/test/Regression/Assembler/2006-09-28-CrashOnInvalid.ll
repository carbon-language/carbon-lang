; Test for PR902.  This program is erroneous, but should not crash llvm-as.
; This tests that a simple error is caught and processed correctly.
; RUN: (llvm-as < %s) 2>&1 | grep 'FP constant invalid for type'
void %test() {
  add int 1, 2.0
  ret void
}

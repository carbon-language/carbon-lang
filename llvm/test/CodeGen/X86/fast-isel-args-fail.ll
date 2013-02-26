; RUN: llc < %s -fast-isel -verify-machineinstrs -mtriple=x86_64-apple-darwin10
; Requires: Asserts

; Previously, this would cause an assert.
define i31 @t1(i31 %a, i31 %b, i31 %c) {
entry:
  %add = add nsw i31 %b, %a
  %add1 = add nsw i31 %add, %c
  ret i31 %add1
}

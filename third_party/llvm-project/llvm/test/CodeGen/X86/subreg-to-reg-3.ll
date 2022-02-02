; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s

; CHECK: imull

; Don't eliminate or coalesce away the explicit zero-extension!

define i64 @foo(i64 %a) {
  %b = mul i64 %a, 7823
  %c = and i64 %b, 4294967295
  %d = add i64 %c, 1
  ret i64 %d
}

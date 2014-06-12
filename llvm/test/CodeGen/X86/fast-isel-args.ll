; RUN: llc < %s -fast-isel -fast-isel-abort -fast-isel-abort-args -verify-machineinstrs -mtriple=x86_64-apple-darwin10

; Just make sure these don't abort when lowering the arguments.
define i32 @t1(i32 %a, i32 %b, i32 %c) {
entry:
  %add = add nsw i32 %b, %a
  %add1 = add nsw i32 %add, %c
  ret i32 %add1
}

define i64 @t2(i64 %a, i64 %b, i64 %c) {
entry:
  %add = add nsw i64 %b, %a
  %add1 = add nsw i64 %add, %c
  ret i64 %add1
}

define i64 @t3(i32 %a, i64 %b, i32 %c) {
entry:
  %conv = sext i32 %a to i64
  %add = add nsw i64 %conv, %b
  %conv1 = sext i32 %c to i64
  %add2 = add nsw i64 %add, %conv1
  ret i64 %add2
}

; RUN: llc < %s -O0 -arm-fast-isel -fast-isel-abort -mtriple=armv7-apple-darwin

; Very basic fast-isel functionality.

define i32 @add(i32 %a, i32 %b) nounwind ssp {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr
  store i32 %b, i32* %b.addr
  %tmp = load i32* %a.addr
  %tmp1 = load i32* %b.addr
  %add = add nsw i32 %tmp, %tmp1
  ret i32 %add
}

; RUN: llc -march=x86 -mcpu=pentium4 -mtriple=i686-none-linux < %s
; PR11314

; Make sure the scheduler's hack to insert artificial dependencies to optimize
; two-address instruction scheduling doesn't interfere with the scheduler's
; hack to model call sequences as artificial physical registers.

define inreg { i64, i64 } @sscanf(i32 inreg %base.1.i) nounwind {
entry:
  %conv38.i92.i = sext i32 %base.1.i to i64
  %rem.i93.i = urem i64 10, %conv38.i92.i
  %div.i94.i = udiv i64 10, %conv38.i92.i
  %a = insertvalue { i64, i64 } undef, i64 %rem.i93.i, 0
  %b = insertvalue { i64, i64 } %a, i64 %div.i94.i, 1
  ret { i64, i64 } %b
}

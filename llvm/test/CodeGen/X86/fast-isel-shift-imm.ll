; RUN: llc < %s -march=x86 -O0 | grep {sarl	\$80, %e}
; PR3242

define void @foo(i32 %x, i32* %p) nounwind {
  %y = ashr i32 %x, 50000
  store i32 %y, i32* %p
  ret void
}

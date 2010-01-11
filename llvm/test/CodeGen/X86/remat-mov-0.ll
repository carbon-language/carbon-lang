; RUN: llc < %s -march=x86-64 | grep {xorl	%edi, %edi} | count 4

; CodeGen should remat the zero instead of spilling it.

declare void @foo(i64 %p)

define void @bar() nounwind {
  call void @foo(i64 0)
  call void @foo(i64 0)
  call void @foo(i64 0)
  call void @foo(i64 0)
  ret void
}

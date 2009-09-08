; RUN: llc < %s -march=x86-64 | grep {leal.*-2(\[%\]rdi,\[%\]rdi)}

define i32 @foo(i32 %x) nounwind readnone {
  %t0 = shl i32 %x, 1
  %t1 = add i32 %t0, -2
  ret i32 %t1
}


; Test that appending linkage works correctly.

; RUN: echo "%X = appending global [1x int] [int 8]" | llvm-as > %t.2.bc
; RUN: llvm-as < %s > %t.1.bc
; RUN: llvm-link %t.[12].bc | llvm-dis | grep 7 | grep 4 | grep 8

%X = appending global [2 x int] [int 7, int 4]

%Y = global int* getelementptr ([2 x int]* %X, long 0, long 0)

void %foo(long %V) {
  %Y = getelementptr [2 x int]* %X, long 0, long %V
  ret void
}

; RUN: llvm-as < %s | opt -funcresolve | llvm-dis | not grep declare

%Table = constant int(...)* %foo

%Table2 = constant [1 x int(...)* ] [ int(...)* %foo ]

declare int %foo(...)

int %foo() {
  ret int 0
}

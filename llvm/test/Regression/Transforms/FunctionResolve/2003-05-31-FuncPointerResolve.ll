; RUN: if as < %s | opt -funcresolve | dis | grep declare
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi
%Table = constant int(...)* %foo

%Table2 = constant [1 x int(...)* ] [ int(...)* %foo ]

declare int %foo(...)

int %foo() {
  ret int 0
}

; RUN: llvm-as < %s | llvm-dis | grep 'tail call csretcc'

declare csretcc void %foo({}*, ...)

void %bar() {
  tail call csretcc void({}*, ...)* %foo({}* null, int 0)
  ret void
}

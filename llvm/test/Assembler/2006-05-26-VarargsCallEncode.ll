; RUN: llvm-upgrade < %s | llvm-as | llvm-dis | \
; RUN:    grep {tail call void (\{  \}\\* sret}

declare csretcc void %foo({}*, ...)

void %bar() {
  tail call csretcc void({}*, ...)* %foo({}* null, int 0)
  ret void
}

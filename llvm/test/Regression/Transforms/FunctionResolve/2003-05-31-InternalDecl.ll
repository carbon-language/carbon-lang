; RUN: llvm-as < %s | opt -funcresolve | llvm-dis | grep declare

declare void %test(...)

int %callee() {
  call void(...)* %test(int 5)
  ret int 2
}

internal void %test(int) {
  ret void
}

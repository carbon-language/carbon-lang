; RUN: llvm-as < %s | opt -funcresolve | dis | not grep declare

declare void %test(int)

int %callee(int %X) {
  call void %test(int %X)
  ret int 2
}

internal void %test(sbyte) {
  ret void
}

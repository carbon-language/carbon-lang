; RUN: as < %s | opt -funcresolve | dis | not grep declare

declare void %test(...)

int %callee() {
  call void(...)* %test(int 5)
  ret int 2
}

internal void %test(int) {
  ret void
}

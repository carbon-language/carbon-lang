; RUN: if as < %s | opt -funcresolve | dis | grep declare
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

declare void %test(...)

int %callee() {
  call void(...)* %test(int 5)
  ret int 2
}

internal void %test(int) {
  ret void
}

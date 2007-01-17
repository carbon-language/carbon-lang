; This situation can occur due to the funcresolve pass.
;
; RUN: llvm-upgrade < %s | llvm-as | opt -raiseallocs | llvm-dis | not grep call

declare void %free(sbyte*)

void %test(int *%P) {
  call void(int*)* cast (void(sbyte*)* %free to void(int*)*)(int* %P)
  ret void
}

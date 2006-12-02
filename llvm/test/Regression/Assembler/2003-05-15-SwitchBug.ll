; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f

; Check minimal switch statement

void %test(int %X) {
  switch int %X, label %dest []
dest:
  ret void
}

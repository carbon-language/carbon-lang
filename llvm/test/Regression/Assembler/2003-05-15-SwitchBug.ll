; RUN: llvm-as < %s -o /dev/null -f



void %test(int %X) {
  switch int %X, label %dest []
dest:
  ret void
}

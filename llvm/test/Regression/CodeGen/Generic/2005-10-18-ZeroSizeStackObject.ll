; RUN: llvm-as < %s | llc

void %test() {
  %X = alloca {}
  ret void
}

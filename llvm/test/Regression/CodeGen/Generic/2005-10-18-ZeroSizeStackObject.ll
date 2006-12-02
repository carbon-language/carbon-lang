; RUN: llvm-upgrade < %s | llvm-as | llc

void %test() {
  %X = alloca {}
  ret void
}

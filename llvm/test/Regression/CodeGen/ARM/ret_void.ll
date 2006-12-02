; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm
void %test() {
  ret void
}

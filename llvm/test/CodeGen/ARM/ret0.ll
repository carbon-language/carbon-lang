; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm
int %test() {
  ret int 0
}

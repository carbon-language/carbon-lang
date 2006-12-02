; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm
int %test(int %a1, int %a2) {
  ret int %a2
}

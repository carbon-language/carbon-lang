; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm
int %test(int %a1) {
  ret int %a1
}

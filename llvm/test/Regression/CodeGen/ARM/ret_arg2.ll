; RUN: llvm-as < %s | llc -march=arm
int %test(int %a1, int %a2) {
  ret int %a2
}

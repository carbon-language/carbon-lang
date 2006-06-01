; RUN: llvm-as < %s | llc -march=arm
int %test(int %a1) {
  ret int %a1
}

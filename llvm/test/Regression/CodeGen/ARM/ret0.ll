; RUN: llvm-as < %s | llc -march=arm
int %test() {
  ret int 0
}

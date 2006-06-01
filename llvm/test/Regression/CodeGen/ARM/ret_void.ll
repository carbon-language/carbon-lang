; RUN: llvm-as < %s | llc -march=arm
void %test() {
  ret void
}

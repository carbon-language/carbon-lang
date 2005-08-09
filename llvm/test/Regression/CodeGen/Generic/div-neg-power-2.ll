; RUN: llvm-as < %s | llc

int %test(int %X) {
  %Y = div int %X, -2
  ret int %Y
}

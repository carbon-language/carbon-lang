; RUN: llvm-as < %s | llc -march=c

int %foo() {
  ret int and (int 123456, int cast (int()* %foo to int))
}

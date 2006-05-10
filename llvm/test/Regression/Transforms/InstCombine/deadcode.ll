; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep 'ret int %A'

int %test(int %A) {
  %X = or bool false, false
  br bool %X, label %T, label %C
T:
  %B = add int %A, 1
  br label %C
C:
  %C = phi int [%B, %T], [%A, %0]
  ret int %C
}

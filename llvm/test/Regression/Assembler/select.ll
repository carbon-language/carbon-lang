; RUN: llvm-as < %s -o /dev/null -f



int %test(bool %C, int %V1, int %V2) {
  %X = select bool true, bool false, bool true
  %V = select bool %X, int %V1, int %V2
  ret int %V
}

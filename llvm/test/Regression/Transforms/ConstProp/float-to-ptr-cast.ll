; RUN: llvm-as < %s | opt -constprop | llvm-dis | grep -F 'ret int* null' | wc -l | grep 2
int* %test1() {
  %X = cast float 0.0 to int*
  ret int* %X
}

int* %test2() {
  ret int* cast (float 0.0 to int*)
}

; RUN: llvm-upgrade < %s | llvm-as | opt -constprop | llvm-dis | \
; RUN:    grep -F {ret i32* null} | wc -l | grep 2

int* %test1() {
  %X = cast float 0.0 to int*
  ret int* %X
}

int* %test2() {
  ret int* cast (float 0.0 to int*)
}

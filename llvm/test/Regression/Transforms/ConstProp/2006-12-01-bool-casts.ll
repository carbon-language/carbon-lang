; RUN: llvm-as < %s | opt -constprop | llvm-dis | grep 'ret int -1' &&
; RUN: llvm-as < %s | opt -constprop | llvm-dis | grep 'ret uint 1'

int %test1() {
  %A = sext bool true to int
  ret int %A
}

uint %test2() {
  %A = zext bool true to uint
  ret uint %A
}


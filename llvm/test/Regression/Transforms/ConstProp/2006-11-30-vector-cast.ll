; RUN: llvm-as < %s | opt -constprop | llvm-dis | grep 4294967295 &&
; RUN: llvm-as < %s | opt -constprop | llvm-dis | not grep zeroinitializer

< 4 x uint> %test() {
  %tmp40 = bitcast <2 x long> bitcast (<4 x int> < int 0, int 0, int -1, int 0 > to <2 x long>) to <4 x uint>
  ret <4 x uint> %tmp40
}

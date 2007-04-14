; RUN: llvm-upgrade < %s | llvm-as | opt -constprop | llvm-dis | \
; RUN:   grep {i32 -1}
; RUN: llvm-upgrade < %s | llvm-as | opt -constprop | llvm-dis | \
; RUN:   not grep zeroinitializer

< 4 x uint> %test() {
  %tmp40 = bitcast <2 x long> bitcast (<4 x int> < int 0, int 0, int -1, int 0 > to <2 x long>) to <4 x uint>
  ret <4 x uint> %tmp40
}

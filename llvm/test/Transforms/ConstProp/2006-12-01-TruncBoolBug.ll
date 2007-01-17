; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep 'ret i1 false'
bool %test() {
  %X = trunc uint 320 to bool
  ret bool %X
}

; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep 'ret bool false'
bool %test() {
  %X = trunc uint 320 to bool
  ret bool %X
}

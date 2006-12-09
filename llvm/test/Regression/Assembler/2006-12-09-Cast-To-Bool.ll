; RUN: llvm-upgrade < %s | llvm-as | llvm-dis | grep bitcast
bool %main(int %X) {
  %res = cast bool true to bool
  ret bool %res
}

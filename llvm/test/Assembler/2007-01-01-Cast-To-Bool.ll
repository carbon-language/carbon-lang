; RUN: llvm-upgrade < %s | llvm-as | llvm-dis | grep {icmp ne}
bool %main(int %X) {
  %res = cast bool true to bool
  ret bool %res
}

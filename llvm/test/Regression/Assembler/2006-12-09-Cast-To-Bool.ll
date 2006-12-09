; RUN llvm-upgrade < %s | llvm-as
bool %main(int %X) {
  %res = cast bool true to bool
  ret bool %res
}

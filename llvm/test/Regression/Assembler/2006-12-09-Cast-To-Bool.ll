; RUN: llvm-as < %s | llvm-dis | grep bitcast
define bool %main(i32 %X) {
  %res = bitcast bool true to bool
  ret bool %res
}

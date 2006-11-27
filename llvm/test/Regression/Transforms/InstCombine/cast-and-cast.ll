; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep bitcast
bool %test1(uint %val) {
  %t1 = bitcast uint %val to int 
  %t2 = and int %t1, 1
  %t3 = trunc int %t2 to bool
  ret bool %t3
}

short %test1(uint %val) {
  %t1 = bitcast uint %val to int 
  %t2 = and int %t1, 1
  %t3 = trunc int %t2 to short
  ret short %t3
}

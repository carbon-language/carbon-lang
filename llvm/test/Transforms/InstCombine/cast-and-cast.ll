; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   not grep bitcast

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

; RUN: llvm-upgrade < %s | llvm-as | llvm-dis | not grep bitcast

int %test1() {
   ret int bitcast(float 0x400D9999A0000000 to int)
}

float %test2() {
  ret float bitcast(int 17 to float)
}

long %test3() {
  ret long bitcast (double 0x400921FB4D12D84A to long)
}

double %test4() {
  ret double bitcast (long 42 to double)
}


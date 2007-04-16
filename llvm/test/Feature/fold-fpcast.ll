; RUN: llvm-upgrade < %s | llvm-as | llvm-dis | not grep bitcast

int %test1() {
   ret int bitcast(float 3.7 to int)
}

float %test2() {
  ret float bitcast(int 17 to float)
}

long %test3() {
  ret long bitcast (double 3.1415926 to long)
}

double %test4() {
  ret double bitcast (long 42 to double)
}


; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86-64
; PR1033

long %test1(double %t) {
  %u = bitcast double %t to long
  ret long %u
}

double %test2(long %t) {
  %u = bitcast long %t to double
  ret double %u
}

int %test3(float %t) {
  %u = bitcast float %t to int
  ret int %u
}

float %test4(int %t) {
  %u = bitcast int %t to float
  ret float %u
}

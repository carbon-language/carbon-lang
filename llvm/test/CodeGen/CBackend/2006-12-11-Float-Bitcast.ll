; RUN: llvm-upgrade < %s | llvm-as | llc -march=c | \
; RUN:   grep __BITCAST | wc -l | grep 14

int %test1(float %F) {
   %X = bitcast float %F to int
   ret int %X
}

float %test2(int %I) {
  %X = bitcast int %I to float
  ret float %X
}

long %test3(double %D) {
  %X = bitcast double %D to long
  ret long %X
}

double %test4(long %L) {
  %X = bitcast long %L to double
  ret double %X
}

double %test5(double %D) {
  %X = bitcast double %D to double
  %Y = add double %X, 2.0
  %Z = bitcast double %Y to long
  %res = bitcast long %Z to double
  ret double %res
}

float %test6(float %F) {
  %X = bitcast float %F to float
  %Y = add float %X, 2.0
  %Z = bitcast float %Y to int
  %res = bitcast int %Z to float
  ret float %res
}

int %main(int %argc, sbyte** %argv) {
  %a = call int %test1(float 3.1415926)
  %b = call float %test2(int %a)
  %c = call long %test3(double 3.1415926)
  %d = call double %test4(long %c)
  %e = call double %test5(double 7.0)
  %f = call float %test6(float 7.0)
  ret int %a
}

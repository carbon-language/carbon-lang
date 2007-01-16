; RUN: llvm-as < %s | llvm-dis | not grep bitcast

define i60 %test1() {
   ret i60 fptoui(float 3.7 to i60)
}

define float %test2() {
  ret float uitofp(i60 17 to float)
}

define i64 %test3() {
  ret i64 bitcast (double 3.1415926 to i64)
}

define double %test4() {
  ret double bitcast (i64 42 to double)
}


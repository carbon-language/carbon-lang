; RUN: llvm-as < %s | opt -globalsmodref-aa -load-vn -gcse | llvm-dis | not grep load

; This test requires the use of previous analyses to determine that 
; doesnotmodX does not modify X (because 'sin' doesn't).

%X = internal global int 4

declare double %sin(double)

int %test(int *%P) {
  store int 12,  int* %X
  call double %doesnotmodX(double 1.0)
  %V = load int* %X
  ret int %V
}

double %doesnotmodX(double %V) {
  %V2 = call double %sin(double %V)
  ret double %V2
}

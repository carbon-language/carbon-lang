; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call.*pow'

declare double %pow(double,double)
%fpstorage = global double 5.0

implementation   ; Functions:

int %main () {
  %fpnum = load double* %fpstorage;
  %one = call double %pow(double 1.0, double %fpnum)
  %two = call double %pow(double %one, double 0.5)
  %three = call double %pow(double %two, double 1.0)
  %four = call double %pow(double %three, double -1.0)
  %five = call double %pow(double %four, double 0.0)
  %result = cast double %five to int
  ret int %result
}


double %test(double* %DP) {
	%D = load double* %DP
	%V = add double %D, 1.0
	%W = sub double %V, %V
	%X = mul double %W, %W
	%Y = div double %X, %X
	%Z = rem double %Y, %Y
	store double %Z, double* %DP
	ret double %Z
}

int %main() { 
  %X = alloca double
  call double %test(double* %X)
  ret int 0 
}

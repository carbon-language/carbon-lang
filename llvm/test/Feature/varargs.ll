


int %test(int %X, ...) {
	%ap = alloca sbyte*
	; This is not a legal testcase, it just shows the syntax for va_arg
	%tmp = va_arg sbyte** %ap, int 
	ret int %tmp
}

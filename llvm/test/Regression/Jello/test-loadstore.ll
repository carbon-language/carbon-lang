
void %test(sbyte* %P, short* %P, int* %P) {
	%V = load sbyte* %P
	store sbyte %V, sbyte* %P

	%V = load short* %P
	store short %V, short* %P

	%V = load int* %P
	store int %V, int* %P
	ret void
}

void %main() {
	%A = alloca sbyte
	%B = alloca short
	%C = alloca int
	call void %test(sbyte* %A, short* %B, int* %C)
	ret void
}

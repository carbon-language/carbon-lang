// RUN: %clang_cc1 -fsyntax-only -Wunused-variable -verify %s

struct s0 {
	unsigned int	i;
};

int proto(int a, int b);

void f0(void) {
	int	a __attribute__((unused)),
		b; // expected-warning{{unused}}
	return;
}

void f1(void) {
	int	i;
	(void)sizeof(i);
	return;
}

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

// PR5933
int f2() {
  int X = 4;  // Shouldn't have a bogus 'unused variable X' warning.
  return Y + X; // expected-error {{use of undeclared identifier 'Y'}}
}

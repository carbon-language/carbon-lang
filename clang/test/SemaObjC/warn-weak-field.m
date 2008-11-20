// RUN: clang -fsyntax-only -fobjc-gc -verify %s

struct S {
	__weak id w; // expected-warning {{__weak attribute cannot be specified on a field declaration}}
	__strong id p1;
};

int main ()
{
	struct I {
        __weak id w1;  // expected-warning {{__weak attribute cannot be specified on a field declaration}}
	};
}

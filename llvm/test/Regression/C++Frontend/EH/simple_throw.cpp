// Test throwing a constant int
#include <stdio.h>

static void foo() { throw 5; }
int main() {
	try {
		foo();
	} catch (...) {
		printf("All ok\n");
		return 0;
	}
	return 1;
}

// This testcase doesn't actually DO any EH
#include <stdio.h>

static void foo() {}
int main() {
	try {
		foo();
	} catch(...) {
		return 1;
	}
	printf("All ok\n");
	return 0;
}

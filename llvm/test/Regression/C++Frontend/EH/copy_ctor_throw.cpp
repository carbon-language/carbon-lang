/* Test for throwing an exception from the copy ctor of the exception object 
 * invoked while building an exception.
 */
#include <stdio.h>

struct foo {
  foo() {}
  foo(const foo &F) { throw 1; }
};

int main() {
	try {
		foo f;
		throw f;
	} catch (int i) {
		printf("Success!\n");
		return 0;
	} catch (foo &f) {
		printf("Failure: caught a foo!\n");
		return 1;
	} catch (...) {
		printf("Failure: caught something else!\n");
		return 1;
	}
}

// RUN: clang -warn-dead-stores -verify %s

void x() {
  int k, y;
	int abc=1;
	long idx=abc+3*5; // expected-warning {{value stored to variable is never used}}
}

void a(void *b) {
 char *c = (char*)b; // no-warning
 char *d = b+1; // expected-warning {{value stored to variable is never used}}
 printf("%s", c);
}

void z() {
	int r;
	if ((r = f()) != 0) { // no-warning
		int y = r; // no-warning
		printf("the error is: %d\n", y);
	}
}

// RUN: clang -warn-dead-stores -verify %s
void x() {
  int k, y;
	int abc=1;
	long idx=abc+3*5; // expected-warning {{value stored to variable is never used}}
}

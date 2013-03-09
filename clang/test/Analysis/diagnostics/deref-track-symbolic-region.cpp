// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-output=text -verify %s

struct S {
  int *x;
  int y;
};

S &getSomeReference();
void test(S *p) {
  S &r = *p;   //expected-note {{'r' initialized here}}
  if (p) return;
               //expected-note@-1{{Taking false branch}}
               //expected-note@-2{{Assuming 'p' is null}}
  r.y = 5; // expected-warning {{Access to field 'y' results in a dereference of a null pointer (loaded from variable 'r')}}
           // expected-note@-1{{Access to field 'y' results in a dereference of a null pointer (loaded from variable 'r')}}
}

void testRefParam(int *ptr) {
	int &ref = *ptr; // expected-note {{'ref' initialized here}}
	if (ptr)
    // expected-note@-1{{Assuming 'ptr' is null}}
    // expected-note@-2{{Taking false branch}}
		return;

	extern void use(int &ref);
	use(ref); // expected-warning{{Forming reference to null pointer}}
            // expected-note@-1{{Forming reference to null pointer}}
}
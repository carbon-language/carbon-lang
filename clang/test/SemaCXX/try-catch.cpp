// RUN: clang -fsyntax-only -verify %s

struct A; // expected-note 3 {{forward declaration of 'struct A'}}

void f()
{
  try {
  } catch(int i) { // expected-note {{previous definition}}
    int j = i;
    int i; // expected-error {{redefinition of 'i'}}
  } catch(float i) {
  } catch(void v) { // expected-error {{cannot catch incomplete type 'void'}}
  } catch(A a) { // expected-error {{cannot catch incomplete type 'struct A'}}
  } catch(A *a) { // expected-error {{cannot catch pointer to incomplete type 'struct A'}}
  } catch(A &a) { // expected-error {{cannot catch reference to incomplete type 'struct A'}}
  } catch(...) {
    int j = i; // expected-error {{use of undeclared identifier 'i'}}
  }

  try {
  } catch(...) { // expected-error {{catch-all handler must come last}}
  } catch(int) {
  }
}

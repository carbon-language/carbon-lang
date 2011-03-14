// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR9463
double *end;
void f() {
  {
    int end = 0;
    goto end;
    end = 1;
  }

 end:
  return;
}

void g() {
  end = 1; // expected-error{{assigning to 'double *' from incompatible type 'int'}}
}

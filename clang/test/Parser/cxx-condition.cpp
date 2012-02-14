// RUN: %clang_cc1 -fsyntax-only -verify %s

void f() {
  int a;
  while (a) ;
  while (int x) ; // expected-error {{expected '=' after declarator}}
  while (float x = 0) ;
  if (const int x = a) ; // expected-warning{{empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
  switch (int x = a+10) {}
  for (; int x = ++a; ) ;
}

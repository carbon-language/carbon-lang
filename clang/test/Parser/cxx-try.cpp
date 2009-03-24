// RUN: clang-cc -fsyntax-only -verify %s

void f()
{
  try {
    ;
  } catch(int i) {
    ;
  } catch(...) {
  }
}

void g()
{
  try; // expected-error {{expected '{'}}

  try {}
  catch; // expected-error {{expected '('}}

  try {}
  catch (...); // expected-error {{expected '{'}}

  try {}
  catch {} // expected-error {{expected '('}}
}

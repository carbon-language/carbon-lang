// RUN: clang %s -fsyntax-only -verify -pedantic-errors

int a() {int p; *(1 ? &p : (void*)(0 && (a(),1))) = 10;} // expected-error {{null pointer expression is not an integer constant expression (but is allowed as an extension)}} // expected-note{{C does not permit evaluated commas in an integer constant expression}}

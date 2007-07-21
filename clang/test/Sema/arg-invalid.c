// RUN: clang %s -parse-ast-check

void bar (void *); 
void f11 (z)       // expected-error {{may not have 'void' type}}
void z; 
{ bar (&z); }

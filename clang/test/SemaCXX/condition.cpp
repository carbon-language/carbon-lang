// RUN: clang -fsyntax-only -verify %s 

void test() {
  int x;
  if (x) ++x;
  if (int x=0) ++x;

  typedef int arr[10];
  while (arr x=0) ; // expected-error: {{an array type is not allowed here}} expected-error: {{initialization with "{...}" expected for array}}
  while (int f()=0) ; // expected-error: {{a function type is not allowed here}}

  struct S {} s;
  if (s) ++x; // expected-error: {{expression must have bool type (or be convertible to bool) ('struct S' invalid)}}
  while (struct S x=s) ; // expected-error: {{expression must have bool type (or be convertible to bool) ('struct S' invalid)}}
  switch (s) {} // expected-error: {{statement requires expression of integer type ('struct S' invalid)}}

  while (struct S {} x=0) ; // expected-error: {{types may not be defined in conditions}} expected-error: {{incompatible type}} expected-error: {{expression must have bool type}}
  while (struct {} x=0) ; // expected-error: {{types may not be defined in conditions}} expected-error: {{incompatible type}} expected-error: {{expression must have bool type}}
  switch (enum {E} x=0) ; // expected-error: {{types may not be defined in conditions}}

  if (int x=0) { // expected-error: {{previous definition is here}}
    int x;  // expected-error: {{redefinition of 'x'}}
  }
  else
    int x;  // expected-error: {{redefinition of 'x'}}
  while (int x=0) int x; // expected-error: {{redefinition of 'x'}} expected-error: {{previous definition is here}}
  while (int x=0) { int x; } // expected-error: {{redefinition of 'x'}} expected-error: {{previous definition is here}}
  for (int x; int x=0; ) ; // expected-error: {{redefinition of 'x'}} expected-error: {{previous definition is here}}
  for (int x; ; ) int x; // expected-error: {{redefinition of 'x'}} expected-error: {{previous definition is here}}
  for (; int x=0; ) int x; // expected-error: {{redefinition of 'x'}} expected-error: {{previous definition is here}}
  for (; int x=0; ) { int x; } // expected-error: {{redefinition of 'x'}} expected-error: {{previous definition is here}}
  switch (int x=0) { default: int x; } // expected-error: {{redefinition of 'x'}} expected-error: {{previous definition is here}}
}

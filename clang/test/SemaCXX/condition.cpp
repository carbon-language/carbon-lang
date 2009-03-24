// RUN: clang-cc -fsyntax-only -verify %s 

void test() {
  int x;
  if (x) ++x;
  if (int x=0) ++x;

  typedef int arr[10];
  while (arr x=0) ; // expected-error {{an array type is not allowed here}} expected-error {{initialization with '{...}' expected for array}}
  while (int f()=0) ; // expected-error {{a function type is not allowed here}}

  struct S {} s;
  if (s) ++x; // expected-error {{value of type 'struct S' is not contextually convertible to 'bool'}}
  while (struct S x=s) ; // expected-error {{value of type 'struct S' is not contextually convertible to 'bool'}}
  do ; while (s); // expected-error {{value of type 'struct S' is not contextually convertible to 'bool'}}
  for (;s;) ; // expected-error {{value of type 'struct S' is not contextually convertible to 'bool'}}
  switch (s) {} // expected-error {{statement requires expression of integer type ('struct S' invalid)}}

  while (struct S {} x=0) ; // expected-error {{types may not be defined in conditions}} expected-error {{cannot initialize 'x' with an rvalue of type 'int'}} expected-error {{value of type 'struct S' is not contextually convertible to 'bool'}}
  while (struct {} x=0) ; // expected-error {{types may not be defined in conditions}} expected-error {{cannot initialize 'x' with an rvalue of type 'int'}} expected-error {{value of type 'struct <anonymous>' is not contextually convertible to 'bool'}}
  switch (enum {E} x=0) ; // expected-error {{types may not be defined in conditions}} expected-error {{incompatible type}}

  if (int x=0) { // expected-note 2 {{previous definition is here}}
    int x;  // expected-error {{redefinition of 'x'}}
  }
  else
    int x;  // expected-error {{redefinition of 'x'}}
  while (int x=0) int x; // expected-error {{redefinition of 'x'}} expected-note {{previous definition is here}}
  while (int x=0) { int x; } // expected-error {{redefinition of 'x'}} expected-note {{previous definition is here}}
  for (int x; int x=0; ) ; // expected-error {{redefinition of 'x'}} expected-note {{previous definition is here}}
  for (int x; ; ) int x; // expected-error {{redefinition of 'x'}} expected-note {{previous definition is here}}
  for (; int x=0; ) int x; // expected-error {{redefinition of 'x'}} expected-note {{previous definition is here}}
  for (; int x=0; ) { int x; } // expected-error {{redefinition of 'x'}} expected-note {{previous definition is here}}
  switch (int x=0) { default: int x; } // expected-error {{redefinition of 'x'}} expected-note {{previous definition is here}}
}

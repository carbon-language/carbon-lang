// RUN: clang %s -fsyntax-only -verify -pedantic

enum e {A, 
        B = 42LL << 32,        // expected-warning {{ISO C restricts enumerator values to range of 'int'}}
      C = -4, D = 12456 };

enum f { a = -2147483648, b = 2147483647 }; // ok.

enum g {  // too negative
   c = -2147483649,         // expected-warning {{ISO C restricts enumerator values to range of 'int'}}
   d = 2147483647 };
enum h { e = -2147483648, // too pos
   f = 2147483648           // expected-warning {{ISO C restricts enumerator values to range of 'int'}}
}; 

// minll maxull
enum x                      // expected-warning {{enumeration values exceed range of largest integer}}
{ y = -9223372036854775807LL-1,  // expected-warning {{ISO C restricts enumerator values to range of 'int'}}
z = 9223372036854775808ULL };    // expected-warning {{ISO C restricts enumerator values to range of 'int'}}

int test() {
  return sizeof(enum e) ;
}

enum gccForwardEnumExtension ve; // expected-error {{variable has incomplete type 'enum gccForwardEnumExtension'}} expected-warning{{ISO C forbids forward references to 'enum' types}}

int test2(int i)
{
  ve + i;
}

// PR2020
union u0;    // expected-error {{previous use is here}}
enum u0 { U0A }; // expected-error {{error: use of 'u0' with tag type that does not match previous declaration}}


// rdar://6095136
extern enum some_undefined_enum ve2; // expected-warning {{ISO C forbids forward references to 'enum' types}}

void test4() {
  for (; ve2;) // expected-error {{statement requires expression of scalar type}}
    ;
  (_Bool)ve2;  // expected-error {{arithmetic or pointer type is required}}

  for (; ;ve2)
    ;
  (void)ve2;
  ve2;         // expected-warning {{expression result unused}}
}

// PR2416
enum someenum {};  // expected-warning {{use of empty enum extension}}


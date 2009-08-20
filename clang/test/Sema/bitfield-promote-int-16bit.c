// RUN: clang-cc -fsyntax-only -verify %s -triple pic16-unknown-unknown

// Check that int-sized unsigned bit-fields promote to unsigned int
// on targets where sizeof(unsigned short) == sizeof(unsigned int)

enum E { ec1, ec2, ec3 };
struct S {
  enum E          e : 16;
  unsigned short us : 16;
  unsigned long ul1 :  8;
  unsigned long ul2 : 16;
} s;

__typeof(s.e + s.e) x_e;
unsigned x_e;

__typeof(s.us + s.us) x_us;
unsigned x_us;

__typeof(s.ul1 + s.ul1) x_ul1;
signed x_ul1;

__typeof(s.ul2 + s.ul2) x_ul2;
unsigned x_ul2;


Compiler-RT
================================

This directory and its subdirectories contain source code for the compiler
support routines.

Compiler-RT is open source software. You may freely distribute it under the
terms of the license agreement found in LICENSE.txt.

================================

This is a replacment library for libgcc.  Each function is contained
in its own file.  Each function has a corresponding unit test under
test/Unit.

A rudimentary script to test each file is in the file called
test/Unit/test.

Here is the specification for this library:

http://gcc.gnu.org/onlinedocs/gccint/Libgcc.html#Libgcc

Here is a synopsis of the contents of this library:

typedef      int si_int;
typedef unsigned su_int;

typedef          long long di_int;
typedef unsigned long long du_int;

// Integral bit manipulation

di_int __ashldi3(di_int a, si_int b);      // a << b
di_int __ashrdi3(di_int a, si_int b);      // a >> b  arithmetic (sign fill)
di_int __lshrdi3(di_int a, si_int b);      // a >> b  logical    (zero fill)

si_int __clzsi2(si_int a);  // count leading zeroes
si_int __clzdi2(di_int a);  // count leading zeroes
si_int __ctzsi2(si_int a);  // count trailing zeroes
si_int __ctzdi2(di_int a);  // count trailing zeroes

si_int __ffsdi2(di_int a);  // find least significant 1 bit

si_int __paritysi2(si_int a);  // bit parity
si_int __paritydi2(di_int a);  // bit parity

si_int __popcountsi2(si_int a);  // bit population
si_int __popcountdi2(di_int a);  // bit population

// Integral arithmetic

di_int __negdi2    (di_int a);                         // -a
di_int __muldi3    (di_int a, di_int b);               // a * b
di_int __divdi3    (di_int a, di_int b);               // a / b   signed
du_int __udivdi3   (du_int a, du_int b);               // a / b   unsigned
di_int __moddi3    (di_int a, di_int b);               // a % b   signed
du_int __umoddi3   (du_int a, du_int b);               // a % b   unsigned
du_int __udivmoddi4(du_int a, du_int b, du_int* rem);  // a / b, *rem = a % b

//  Integral arithmetic with trapping overflow

si_int __absvsi2(si_int a);           // abs(a)
di_int __absvdi2(di_int a);           // abs(a)

si_int __negvsi2(si_int a);           // -a
di_int __negvdi2(di_int a);           // -a

si_int __addvsi3(si_int a, si_int b);  // a + b
di_int __addvdi3(di_int a, di_int b);  // a + b

si_int __subvsi3(si_int a, si_int b);  // a - b
di_int __subvdi3(di_int a, di_int b);  // a - b

si_int __mulvsi3(si_int a, si_int b);  // a * b
di_int __mulvdi3(di_int a, di_int b);  // a * b

//  Integral comparison: a  < b -> 0
//                       a == b -> 1
//                       a  > b -> 2

si_int __cmpdi2 (di_int a, di_int b);
si_int __ucmpdi2(du_int a, du_int b);

//  Integral / floating point conversion

di_int __fixsfdi(      float a);
di_int __fixdfdi(     double a);
di_int __fixxfdi(long double a);

su_int __fixunssfsi(      float a);
su_int __fixunsdfsi(     double a);
su_int __fixunsxfsi(long double a);

du_int __fixunssfdi(      float a);
du_int __fixunsdfdi(     double a);
du_int __fixunsxfdi(long double a);

float       __floatdisf(di_int a);
double      __floatdidf(di_int a);
long double __floatdixf(di_int a);

float       __floatundisf(du_int a);
double      __floatundidf(du_int a);
long double __floatundixf(du_int a);

//  Floating point raised to integer power

float       __powisf2(      float a, si_int b);  // a ^ b
double      __powidf2(     double a, si_int b);  // a ^ b
long double __powixf2(long double a, si_int b);  // a ^ b

//  Complex arithmetic

//  (a + ib) * (c + id)

      float _Complex __mulsc3( float a,  float b,  float c,  float d);
     double _Complex __muldc3(double a, double b, double c, double d);
long double _Complex __mulxc3(long double a, long double b,
                              long double c, long double d);

//  (a + ib) / (c + id)

      float _Complex __divsc3( float a,  float b,  float c,  float d);
     double _Complex __divdc3(double a, double b, double c, double d);
long double _Complex __divxc3(long double a, long double b,
                              long double c, long double d);

Preconditions are listed for each function at the definition when there are any.
Any preconditions reflect the specification at
http://gcc.gnu.org/onlinedocs/gccint/Libgcc.html#Libgcc.

Assumptions are listed in "int_lib.h", and in individual files.  Where possible
assumptions are checked at compile time.

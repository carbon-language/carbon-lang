// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-apple-darwin9 -fms-extensions %s
typedef int i128 __attribute__((__mode__(TI)));
typedef unsigned u128 __attribute__((__mode__(TI)));

int a[((i128)-1 ^ (i128)-2) == 1 ? 1 : -1];
int a[(u128)-1 > 1LL ? 1 : -1];

// PR5435
__uint128_t b = (__uint128_t)-1;

// PR11916: Support for libstdc++ 4.7
__int128 i = (__int128)0;
unsigned __int128 u = (unsigned __int128)-1;

long long SignedTooBig = 123456789012345678901234567890; // expected-warning {{integer constant is too large for its type}}
__int128_t Signed128 = 123456789012345678901234567890i128;
long long Signed64 = 123456789012345678901234567890i128; // expected-warning {{implicit conversion from '__int128' to 'long long' changes value from 123456789012345678901234567890 to -4362896299872285998}}
unsigned long long UnsignedTooBig = 123456789012345678901234567890; // expected-warning {{integer constant is too large for its type}}
__uint128_t Unsigned128 = 123456789012345678901234567890Ui128;
unsigned long long Unsigned64 = 123456789012345678901234567890Ui128; // expected-warning {{implicit conversion from 'unsigned __int128' to 'unsigned long long' changes value from 123456789012345678901234567890 to 14083847773837265618}}

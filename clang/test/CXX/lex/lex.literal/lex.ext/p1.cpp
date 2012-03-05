// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

void operator "" p31(long double); // expected-warning{{user-defined literal with suffix 'p31' is preempted by C99 hexfloat extension}}
void operator "" _p31(long double);
long double operator "" pi(long double); // expected-warning{{user-defined literals not starting with '_' are reserved by the implementation}}

float hexfloat = 0x1p31; // allow hexfloats

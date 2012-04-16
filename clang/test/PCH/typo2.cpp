// RUN: %clang_cc1 -emit-pch %s -o %t.pch
// RUN: %clang_cc1 -include-pch %t.pch %s -verify

#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

void func(struct Test);  // expected-note{{'Test' declared here}}

#else

::Yest *T;  // expected-error{{did you mean 'Test'}}

#endif
